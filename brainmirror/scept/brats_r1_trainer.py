"""SCEPT v0.9-R1 fault-tolerant checkpointing and validation-selected training.

Storage semantics: two alternating atomic generations, independently mirrored
into caller-provided backup dir. No guarantee of persistence is claimed: the
caller MUST provision a truly persistent, writable external storage volume.
Original BraTS images and .pt checkpoints never enter the public repository.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Callable

import numpy as np
import torch

from .brats_v09_pipeline import require_clean_splits, canonical_hash


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _atomic_bytes(path: Path, source: Path | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            if isinstance(source, bytes):
                f.write(source)
            else:
                with source.open('rb') as inp:
                    shutil.copyfileobj(inp, f, length=1024 * 1024)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class MirroredCheckpointStore:
    """Two crash-tolerant, SHA-verified slots; backup is recovery authority.

    A crash in the middle of replacing either slot cannot corrupt the OTHER
    verified slot. Manifests are published only after both files are durable
    at the filesystem level. External replication durability remains the
    responsibility of the mounted backup provider.
    """
    def __init__(self, local_dir: str | Path, backup_dir: str | Path, *, prefix: str):
        self.local = Path(local_dir).expanduser().resolve()
        self.backup = Path(backup_dir).expanduser().resolve()
        if self.local == self.backup or self.local in self.backup.parents or self.backup in self.local.parents:
            raise ValueError('Backup must be a separate, non-nested storage directory')
        if not prefix or not prefix.replace('_', '').isalnum():
            raise ValueError('Invalid checkpoint prefix')
        self.prefix = prefix
        for root in (self.local, self.backup):
            root.mkdir(parents=True, exist_ok=True)
            if not root.is_dir():
                raise ValueError('Checkpoint root is not a directory')
            probe = root / f'.{prefix}_write_probe'
            _atomic_bytes(probe, b'probe')
            probe.unlink()

    def _paths(self, root, slot):
        return (root / f'{self.prefix}_slot{slot}.pt',
                root / f'{self.prefix}_slot{slot}.json')

    def verified(self):
        """Return verified generations, checking both backup and local.

        If local file is damaged, use the verified backup without silently
        considering a broken copy valid.
        """
        found = []
        for slot in (0, 1):
            p, m = self._paths(self.backup, slot)
            try:
                manifest = json.loads(m.read_text(encoding='utf-8'))
                if manifest['prefix'] != self.prefix or manifest['slot'] != slot:
                    continue
                if manifest['generation'] < 0 or len(manifest['sha256']) != 64:
                    continue
                if p.stat().st_size != manifest['bytes'] or digest_file(p) != manifest['sha256']:
                    continue
                found.append((manifest['generation'], slot, p, manifest))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(found, key=lambda z: z[0], reverse=True)

    def save(self, payload: dict, *, metadata: dict | None = None) -> dict:
        current = self.verified()
        gen = current[0][0] + 1 if current else 0
        slot = gen % 2
        local_file, local_manifest = self._paths(self.local, slot)
        backup_file, backup_manifest = self._paths(self.backup, slot)
        fd, tmp = tempfile.mkstemp(prefix='.torch-', suffix='.pt', dir=self.local)
        os.close(fd)
        try:
            torch.save(payload, tmp)
            # force the created torch.save bytes to disk before publication
            with open(tmp, 'rb') as inp:
                os.fsync(inp.fileno())
            expected = digest_file(Path(tmp))
            size = Path(tmp).stat().st_size
            # New backup is written BEFORE manifest; the other slot remains safe.
            _atomic_bytes(backup_file, Path(tmp))
            _atomic_bytes(local_file, Path(tmp))
            if digest_file(backup_file) != expected or digest_file(local_file) != expected:
                raise OSError('Mirrored backup differs from checkpoint before committing manifest')
            entry = {'schema': 'scept/r1-mirrored-generation/v1', 'prefix': self.prefix,
                     'slot': slot, 'generation': gen, 'bytes': size,
                     'sha256': expected, 'metadata': metadata or {}}
            raw = (json.dumps(entry, sort_keys=True, indent=2) + '\n').encode()
            _atomic_bytes(backup_manifest, raw)
            _atomic_bytes(local_manifest, raw)
            return entry
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def load(self, *, map_location='cpu'):
        candidates = self.verified()
        if not candidates:
            raise FileNotFoundError('No verified checkpoint generation in independent backup')
        errors = []
        for _, slot, source, m in candidates:
            try:
                obj = torch.load(source, map_location=map_location, weights_only=True)
                if not isinstance(obj, dict):
                    raise ValueError('Checkpoint must be a mapping')
                local_file, local_manifest = self._paths(self.local, slot)
                if not local_file.exists() or digest_file(local_file) != m['sha256']:
                    _atomic_bytes(local_file, source)
                _atomic_bytes(local_manifest, (json.dumps(m, sort_keys=True, indent=2) + '\n').encode())
                return obj, m
            except (OSError, ValueError, RuntimeError, EOFError) as exc:
                errors.append(str(exc))
        raise RuntimeError('All verified candidate checkpoint payloads failed load: ' + '; '.join(errors))


@dataclass(frozen=True)
class R1Config:
    epochs: int = 30
    steps_per_epoch: int = 120
    checkpoint_every: int = 20
    validate_every_epochs: int = 1
    patience_evaluations: int = 10
    seed: int = 240216
    amp: bool = True
    max_grad_norm: float = 1.0
    lr: float = 1e-4
    weight_decay: float = 1e-5

    def __post_init__(self):
        if min(self.epochs, self.steps_per_epoch, self.checkpoint_every,
               self.validate_every_epochs, self.patience_evaluations) < 1:
            raise ValueError('Positive epoch, interval and patience required')
        if not 0 < self.lr < 1 or self.max_grad_norm <= 0 or self.weight_decay < 0:
            raise ValueError('Invalid optimizer configuration')


def _capture_rng(py_rng, np_rng):
    return {'python_global': random.getstate(), 'python_sampling': py_rng.getstate(),
            'numpy_global': (np.random.get_state()[0], np.random.get_state()[1].tolist(),
                             int(np.random.get_state()[2]), int(np.random.get_state()[3]),
                             float(np.random.get_state()[4])),
            'numpy_sampling': np_rng.bit_generator.state,
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state, py_rng, np_rng):
    random.setstate(state['python_global'])
    py_rng.setstate(state['python_sampling'])
    s = state['numpy_global']
    np.random.set_state((s[0], np.asarray(s[1], dtype=np.uint32), s[2], s[3], s[4]))
    np_rng.bit_generator.state = state['numpy_sampling']
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(state['torch_cuda']):
            raise ValueError('CUDA device count changed; cannot claim RNG-equivalent resume')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def _move_optimizer_state(optimizer, device):
    for entry in optimizer.state.values():
        for k,v in entry.items():
            if isinstance(v, torch.Tensor):
                entry[k] = v.to(device)


def _init_rng(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    return random.Random(seed), np.random.default_rng(seed)


def run_resumable_training(*, model: torch.nn.Module, splits: dict,
        inventory_sha256: str, architecture: dict, config: R1Config,
        training_store: MirroredCheckpointStore, best_store: MirroredCheckpointStore,
        sample_fn: Callable, validation_fn: Callable, device='cpu', resume=False,
        stop_after_steps: int | None = None, on_progress: Callable | None = None):
    """Run (possibly partial) training. Functions can be tested without MONAI.

    sample_fn(case_id, step, py_rng, np_rng) -> (x,y); validation_fn(model)
    returns {'WT': float, 'TC': float, 'ET': float}, on entire fixed validation
    cohort. The independent TEST cohort must never be passed to callbacks.
    """
    require_clean_splits(splits)
    if len(inventory_sha256)!=64 or not all(c in '0123456789abcdef' for c in inventory_sha256.lower()):
        raise ValueError('Invalid inventory sha256')
    if stop_after_steps is not None and stop_after_steps < 1:
        raise ValueError('stop_after_steps must be positive')
    if training_store.prefix == best_store.prefix and training_store.backup == best_store.backup:
        raise ValueError('Training and best checkpoints need distinct prefixes')
    cfg=asdict(config)
    # Capacity (epochs) can increase, but all scientific sampling/optimiser choices are frozen.
    fingerprint = canonical_hash({'config':{k:v for k,v in cfg.items() if k!='epochs'},
                                   'architecture':architecture, 'splits':splits,
                                   'inventory':inventory_sha256})
    dev=torch.device(device)
    py_rng,np_rng=_init_rng(config.seed)
    model.to(dev)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=config.weight_decay)
    amp_enabled = bool(config.amp and dev.type=='cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=amp_enabled)
    state={'step':0,'last_validated_epoch':0,'best_score':float('-inf'),
           'best_epoch':None,'bad_validations':0,'validation_history':[],
           'last_loss':None}

    if resume:
        saved, manifest = training_store.load(map_location='cpu')
        if saved.get('fingerprint') != fingerprint:
            raise ValueError('Resume fingerprint mismatch: config/splits/data/architecture changed')
        if saved['state']['step'] > config.epochs * config.steps_per_epoch:
            raise ValueError('Resume checkpoint beyond configured training horizon')
        model.load_state_dict(saved['model'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        _move_optimizer_state(optimizer,dev)
        scaler.load_state_dict(saved['scaler'])
        state=saved['state']
        if state['best_epoch'] is not None and not best_store.verified():
            raise RuntimeError('Best validation checkpoint is missing/corrupted; refusing to resume as if model selection were intact')
        _restore_rng(saved['rng'],py_rng,np_rng)
    else:
        if training_store.verified() or best_store.verified():
            raise ValueError('Existing checkpoints detected: use --resume or a fresh destination')

    total=config.epochs*config.steps_per_epoch
    interrupted=False
    def save_checkpoint():
        saved_stamp=training_store.save({'schema':'scept/r1-train-state/v1',
            'fingerprint':fingerprint,'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'rng':_capture_rng(py_rng,np_rng),'state':state},
            metadata={'step':state['step'],'best_epoch':state['best_epoch'],
                      'last_validated_epoch':state['last_validated_epoch']})
        if on_progress is not None:
            on_progress({'event':'checkpoint','step':state['step'],
                         'last_validated_epoch':state['last_validated_epoch'],
                         'best_epoch':state['best_epoch'],
                         'best_score':state['best_score'] if math.isfinite(state['best_score']) else None,
                         'generation':saved_stamp['generation'],
                         'checkpoint_sha256':saved_stamp['sha256']})

    while True:
        step=state['step']
        finished_epoch=(step//config.steps_per_epoch if step else 0)
        due = step>0 and step % config.steps_per_epoch==0 and \
              state['last_validated_epoch'] < finished_epoch and \
              (finished_epoch % config.validate_every_epochs==0 or step==total)
        if due:
            model.eval()
            with torch.inference_mode():
                measured=validation_fn(model)
            if set(measured)!={'WT','TC','ET'} or not all(isinstance(v,(float,int)) and math.isfinite(v) and 0<=v<=1 for v in measured.values()):
                raise ValueError('Require complete finite WT/TC/ET validation Dice; no test-set proxy')
            score=sum(measured.values())/3.
            state['validation_history'].append({'epoch':finished_epoch,'step':step,
                                                 'dice':{k:float(v) for k,v in measured.items()},
                                                 'score':score})
            state['last_validated_epoch']=finished_epoch
            improved=score>state['best_score']+1e-8
            if improved:
                state['best_score']=score
                state['best_epoch']=finished_epoch
                state['bad_validations']=0
                provenance={'schema':'scept/brats2020-checkpoint/v1',
                    'dataset':'BraTS2020_training','monai_version':'1.5.1',
                    'train_ids':sorted(splits['train']),'validation_ids':sorted(splits['validation']),
                    'inventory_sha256':inventory_sha256,'seed':config.seed,
                    'epochs':finished_epoch,'optimization_steps':step,
                    'final_loss':float(state['last_loss']), 'synthetic_only':False,
                    'selection':'full_validation_macro_dice',
                    'validation_dice':{k:float(v) for k,v in measured.items()},
                    'training_config_sha256':fingerprint}
                best_store.save({'architecture':architecture,
                    'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'provenance':provenance},
                    metadata={'epoch':finished_epoch,'score':score,'step':step})
            else:
                state['bad_validations']+=1
            save_checkpoint()
            if state['bad_validations']>=config.patience_evaluations:
                break
            if step>=total:
                break
            continue
        if step>=total:
            break
        if stop_after_steps is not None and step>=stop_after_steps:
            interrupted=True
            break
        case_id=py_rng.choice(splits['train'])
        x,y=sample_fn(case_id,step,py_rng,np_rng)
        if x.ndim!=5 or y.ndim!=5 or x.shape[0]!=y.shape[0] or x.shape[1]!=4 or y.shape[1]!=3:
            raise ValueError('Training sample shapes must be [B,4,D,H,W], [B,3,D,H,W]')
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',enabled=amp_enabled):
            prediction=model(x.to(dev,non_blocking=True))
            from .brats_v09_pipeline import brats_loss
            loss=brats_loss(prediction,y.to(dev,non_blocking=True))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Non-finite training loss; last verified checkpoint retained')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        state['last_loss']=float(loss.detach())
        state['step']+=1
        if state['step'] % config.checkpoint_every==0:
            save_checkpoint()
    if interrupted:
        save_checkpoint()
    return {'schema':'scept/r1-training-report/v1','steps':state['step'],
        'validation_history':state['validation_history'], 'best_score':state['best_score'] if math.isfinite(state['best_score']) else None,
        'best_epoch':state['best_epoch'],'best_available':bool(best_store.verified()),
        'interrupted':interrupted,'completed':not interrupted and
        (state['step']>=total or state['bad_validations']>=config.patience_evaluations),
        'resume_fingerprint':fingerprint,'scientific_evaluation':'NOT_PERFORMED'}
