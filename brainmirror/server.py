"""Local-first BrainMirror experiment server. No patient data is sent remotely."""
import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .model import BrainAE, LAYERS, incoming_edges
from .mri import phantom, local_expand, brain_surface, to_mesh, prepare_pair

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'runs'
CKPTS = ROOT / 'data' / 'checkpoints'
RUNS.mkdir(exist_ok=True, parents=True)
CKPTS.mkdir(exist_ok=True, parents=True)
torch.set_num_threads(min(4, os.cpu_count() or 1))

app = FastAPI(title='BrainMirror | Black Box Observatory', version='0.1.0')
app.mount('/web', StaticFiles(directory=str(ROOT / 'web')), name='web')
lock = threading.RLock()
model = BrainAE().eval()
checkpoint = CKPTS / 'latest.pt'
source_kind = 'synthetic'
subject_id = 'PHANTOM-001'
volume, roi = phantom(1)
real_examples = []
trained = False
training = {'state': 'idle', 'epoch': 0, 'total': 0, 'losses': [], 'error': None, 'source': None}
model_provenance = {'status': 'random weights', 'training_source': None}


def model_digest():
    h = hashlib.sha256()
    for key, weight in sorted(model.state_dict().items()):
        h.update(key.encode('utf-8'))
        h.update(weight.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:20]


def restore():
    global trained, model_provenance
    if not checkpoint.exists():
        return
    try:
        obj = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(obj['state_dict'], strict=True)
        model.eval()
        trained = True
        model_provenance = obj.get('provenance', {'status':'trained checkpoint'})
    except Exception as ex:
        print(f'Checkpoint rejected: {ex}')


restore()


class AnalyzeRequest(BaseModel):
    strength: float = Field(0, ge=0, le=2.5)
    layer: str = 'conv3'
    channel: int = Field(0, ge=0)
    z: int = Field(2, ge=0)
    y: int = Field(2, ge=0)
    x: int = Field(2, ge=0)
    ablate: bool = False


class TrainRequest(BaseModel):
    source: str = 'synthetic'
    epochs: int = Field(8, ge=1, le=80)
    seed: int = Field(42, ge=0, le=99999999)
    phantom_count: int = Field(12, ge=4, le=120)


@app.get('/')
def index():
    return FileResponse(ROOT/'web'/'index.html')


@app.get('/api/state')
def state():
    with lock:
        return {'selected_source': source_kind, 'subject_id': subject_id,
                'real_sample_count': len(real_examples), 'trained': trained,
                'training': training.copy(), 'model_provenance': model_provenance,
                'model_sha256_prefix': model_digest(),
                'layers': LAYERS, 'mode': 'Local CPU/GPU-free demo; no clinical validation'}


@app.post('/api/select_demo')
def select_demo():
    global volume, roi, source_kind, subject_id
    with lock:
        volume, roi = phantom(1)
        source_kind, subject_id = 'synthetic', 'PHANTOM-001'
    return {'ok':True, 'source':source_kind}


@app.post('/api/upload')
async def upload(mri: UploadFile=File(...), mask: UploadFile=File(...)):
    global volume, roi, source_kind, subject_id
    for item in (mri, mask):
        if not item.filename or not item.filename.lower().endswith(('.nii','.nii.gz','.npy')):
            raise HTTPException(400, 'Accept .nii, .nii.gz or .npy only')
    # File upload capped at 128 MiB per file at application level; process in RAM only.
    try:
        vb = await mri.read(128*1024*1024+1)
        mb = await mask.read(128*1024*1024+1)
        if len(vb)>128*1024*1024 or len(mb)>128*1024*1024:
            raise ValueError('Each input must be <=128 MiB')
        v, m = prepare_pair(vb, mri.filename, mb, mask.filename)
    except (ValueError, ImportError, OSError, EOFError) as ex:
        raise HTTPException(400, f'Unable to load aligned MRI/mask: {ex}') from ex
    digest = hashlib.sha256(v.tobytes()).hexdigest()[:10]
    with lock:
        if training['state']=='running':
            raise HTTPException(409,'Training underway; upload after completion')
        real_examples.append(v.copy())
        volume, roi = v, m
        source_kind, subject_id = 'uploaded MRI', f'UPLOAD-{digest}'
    return {'ok':True, 'sample_count':len(real_examples), 'subject_id':subject_id,
            'warning':'ROI warp is exploratory, not a clinical anatomical simulation'}


def training_worker(config: TrainRequest):
    global trained, model_provenance, training
    try:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        with lock:
            if config.source == 'synthetic':
                samples = np.stack([phantom(100+i)[0] for i in range(config.phantom_count)])
            else:
                if not real_examples:
                    raise ValueError('Upload at least one MRI+ROI pair first')
                samples = np.stack(real_examples)
            # Training always starts fresh, so provenance cannot silently carry over.
            model.__init__()
            model.train()
            trained = False
            model_provenance = {'status': 'training', 'training_source':config.source}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        all_x = torch.from_numpy(samples).unsqueeze(1).float()
        for epoch in range(config.epochs):
            order = torch.randperm(len(all_x))
            loss_values = []
            for inds in order.split(4):
                batch = all_x[inds]
                optimizer.zero_grad(set_to_none=True)
                output = model(batch)
                loss = torch.mean((output - batch)**2)
                loss.backward()
                optimizer.step()
                loss_values.append(float(loss.detach()))
            with lock:
                training['epoch'] = epoch + 1
                training['losses'].append(float(np.mean(loss_values)))
        model.eval()
        now = datetime.now(timezone.utc).isoformat()
        provenance = {'status':'trained', 'training_source':config.source,
                      'training_examples':int(len(samples)), 'epochs':config.epochs,
                      'seed':config.seed, 'finished_at_utc':now,
                      'note':'No independent validation set; scientific conclusions not supported by demo training'}
        # Local checkpoint contains weights + non-identifying provenance only.
        payload={'state_dict':model.state_dict(), 'provenance':provenance}
        torch.save(payload, CKPTS / f'model-{model_digest()}.pt')
        torch.save(payload, checkpoint)
        with lock:
            trained = True
            model_provenance = provenance
            training['state']='completed'
    except Exception as ex:
        with lock:
            model.eval()
            trained=False
            training['state']='failed'
            training['error']=str(ex)


@app.post('/api/train')
def train(config: TrainRequest):
    global training
    if config.source not in ('synthetic','uploaded'):
        raise HTTPException(400, 'source must be synthetic or uploaded')
    with lock:
        if training['state']=='running':
            raise HTTPException(409,'Training already running')
        if config.source=='uploaded' and not real_examples:
            raise HTTPException(400,'Upload MRI + ROI mask first')
        training = {'state':'running','epoch':0,'total':config.epochs,'losses':[],
                    'error':None,'source':config.source}
    threading.Thread(target=training_worker, args=(config,), daemon=True).start()
    return {'ok':True, 'message':'Training started; poll /api/state for real progress'}


@app.post('/api/analyze')
def analyze(config: AnalyzeRequest):
    if config.layer not in LAYERS:
        raise HTTPException(400,'Unknown layer')
    dims = LAYERS[config.layer]
    if config.channel>=dims['channels'] or max(config.x,config.y,config.z)>=dims['side']:
        raise HTTPException(400,'Selected neuron index is outside layer dimensions')
    with lock:
        if training['state']=='running':
            raise HTTPException(409,'Training running; inference is locked')
        original, mask = volume.copy(), roi.copy()
        source, case = source_kind, subject_id
        model.eval()
        try:
            changed, changed_mask, qc = local_expand(original,mask,config.strength)
            x0 = torch.from_numpy(original)[None,None]
            x1 = torch.from_numpy(changed)[None,None]
            with torch.no_grad():
                reconstructed_before, a0 = model(x0,capture=True)
                reconstructed_after, a1 = model(x1,capture=True)
                ablated = None
                if config.ablate:
                    ablated, _ = model(x1, capture=True,
                         ablate={'layer':config.layer,'channel':config.channel,
                                 'z':config.z,'y':config.y,'x':config.x})
            g0 = incoming_edges(model, a0,config.layer,config.channel,config.z,config.y,config.x)
            # Every cloud dot is an actual post-ReLU Conv3d tensor element.
            bottleneck_before = a0['conv3'][0].detach().cpu().numpy()
            bottleneck_after = a1['conv3'][0].detach().cpu().numpy()
            cloud = []
            for channel_i in range(bottleneck_after.shape[0]):
                for zi in range(4):
                    for yi in range(4):
                        for xi in range(4):
                            before_val = float(bottleneck_before[channel_i,zi,yi,xi])
                            after_val = float(bottleneck_after[channel_i,zi,yi,xi])
                            cloud.append({'c':channel_i,'z':zi,'y':yi,'x':xi,
                                          'before':before_val,'after':after_val,
                                          'delta':after_val-before_val})

            g1 = incoming_edges(model, a1,config.layer,config.channel,config.z,config.y,config.x)
            pre_delta = g1['preactivation'] - g0['preactivation']
            post_delta = g1['activation'] - g0['activation']
            mse_base = float(torch.mean((reconstructed_before-x0)**2))
            mse_edit = float(torch.mean((reconstructed_after-x1)**2))
            ablation_mean_diff = (float(torch.mean(torch.abs(ablated-reconstructed_after)))
                                  if ablated is not None else None)
            rec_changed_mean_diff = float(torch.mean(torch.abs(reconstructed_after-reconstructed_before)))
        except (ValueError, AssertionError) as ex:
            raise HTTPException(400,str(ex)) from ex
        result = {
            'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'subject_id':case,'source':source,'trained':trained,
            'model_sha256_prefix':model_digest(),'provenance':model_provenance,
            'input_edit':{'roi':'user-provided ROI (synthetic ventricles for demo)',
                          'strength_voxels':config.strength,'qc':qc},
            'unit':{'layer':config.layer,'channel':config.channel,
                    'z':config.z,'y':config.y,'x':config.x},
            'baseline':{'preactivation':g0['preactivation'],'activation':g0['activation']},
            'edited':{'preactivation':g1['preactivation'],'activation':g1['activation']},
            'preactivation_delta':pre_delta,'activation_delta':post_delta,
            'ablation_enabled':config.ablate,
            'ablation_mean_abs_output_change':ablation_mean_diff,
            'mean_abs_reconstruction_change':rec_changed_mean_diff,
            'reconstruction_mse_original':mse_base,
            'reconstruction_mse_edited':mse_edit,
            'incoming_graph':g1,
            'bottleneck_cloud':cloud,
            'notes':[
                'Each displayed node is a real tensor element; each edge is a real Conv3d weight.',
                'Incoming weighted sum + bias is numerically checked against model preactivation.',
                'Neural activation does not establish semantic function or causal mechanism.',
                'Ablation measures effect on this autoencoder reconstruction only.',
                '32^3 resampling and local warp are exploratory, not clinically validated.'
            ]
        }
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        (RUNS / f'{run_id}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
        return {'run_id':run_id,
                'subject_id':case,'source':source,'trained':trained,
                'model_sha256_prefix':result['model_sha256_prefix'],
                'provenance':model_provenance,
                'brain_mesh':brain_surface(changed),
                'roi_mesh':to_mesh(changed_mask),
                'brain_slice':changed[:,:,16].round(3).tolist(),
                'graph':g1, 'cloud':cloud, 'summary':{k:result[k] for k in (
                    'baseline','edited','preactivation_delta','activation_delta',
                    'ablation_mean_abs_output_change','mean_abs_reconstruction_change',
                    'reconstruction_mse_original','reconstruction_mse_edited')},
                'qc':qc}


@app.get('/api/runs')
def runs():
    return {'runs':[p.stem for p in sorted(RUNS.glob('*.json'), reverse=True)[:30]]}


@app.get('/api/runs/{run_id}.json')
def download_run(run_id:str):
    if not re.fullmatch(r'\d{8}T\d{12}Z', run_id):
        raise HTTPException(400,'Invalid run ID')
    filepath=RUNS/f'{run_id}.json'
    if not filepath.exists():
        raise HTTPException(404,'Run not found')
    return FileResponse(filepath, filename=f'brainmirror-{run_id}.json', media_type='application/json')
