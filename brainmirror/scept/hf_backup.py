"""Verified Hugging Face *private* model-repository mirror for SCEPT R1.

A remote mirror, NOT a fake /kaggle/working backup. Store only model/optimizer
checkpoint bytes and the frozen patient-ID split; never upload raw NIfTI volumes.
Manifest is committed after the corresponding checkpoint is uploaded and
independently downloaded/hash-checked. Two slots preserve an older generation.
The Hugging Face account owner manages access, quota and retention.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

from .brats_r1_trainer import _atomic_bytes, digest_file


class HFPrivateBackup:
    """Private remote backup for an existing MirroredCheckpointStore.

    ``api`` and ``downloader`` are injectable to test remote semantics without
    any network or credentials. Actual provider calls use huggingface_hub.
    """

    def __init__(self, repo_id, token, *, path_prefix="scept-v09-r1",
                 api=None, downloader=None, retries=3):
        if not re.fullmatch(r"[\w][\w.-]*/[\w][\w.-]*", repo_id):
            raise ValueError("Expected Hugging Face owner/private-model-repository")
        if not token or not isinstance(token, str):
            raise ValueError("Hugging Face write token is required")
        if not re.fullmatch(r"[\w.-]+", path_prefix) or path_prefix in ('.', '..'):
            raise ValueError("Unsafe remote checkpoint prefix")
        if retries < 1:
            raise ValueError("Retry count must be positive")
        if api is None or downloader is None:
            try:
                from huggingface_hub import HfApi, hf_hub_download
            except ImportError as e:
                raise RuntimeError("Install huggingface_hub>=0.26") from e
            if api is None:
                api = HfApi()
            if downloader is None:
                downloader = hf_hub_download
        self.api = api
        self.download = downloader
        self.repo_id = repo_id
        self.token = token
        self.path_prefix = path_prefix
        self.retries = retries
        info = self.api.repo_info(repo_id=repo_id, repo_type="model", token=token)
        if info.private is not True:
            raise ValueError("Refusing to upload checkpoints to a PUBLIC Hugging Face repo")
        self._published = {}

    def _path(self, fname):
        return f"{self.path_prefix}/{fname}"

    def _remote_file(self, fname, cache_dir):
        return Path(self.download(
            repo_id=self.repo_id, filename=self._path(fname),
            repo_type='model', token=self.token, cache_dir=cache_dir,
            force_download=True,
        ))

    def _remote_bytes(self, fname):
        with tempfile.TemporaryDirectory(prefix='scept-hf-fetch-') as tmp:
            return self._remote_file(fname, tmp).read_bytes()

    def _upload(self, fname, data):
        for attempt in range(self.retries):
            try:
                self.api.upload_file(
                    path_or_fileobj=data, path_in_repo=self._path(fname),
                    repo_id=self.repo_id, repo_type="model", token=self.token,
                    commit_message=f"checkpoint: update {fname}",
                )
                return
            except Exception:
                if attempt + 1 == self.retries:
                    raise
                time.sleep(min(2 ** attempt, 4))

    def probe_writable(self):
        """Prove private remote writes AND independent readback before GPU use."""
        value = b'SCEPT-R1-private-checkpoint-probe-v1\n'
        self._upload('write_probe.txt', value)
        if self._remote_bytes('write_probe.txt') != value:
            raise OSError('Private remote readback differs from uploaded probe')
        return True

    def pin_split(self, split_file: str | Path):
        """Save immutable split and reject even one byte of drift on reuse."""
        value = Path(split_file).read_bytes()
        json.loads(value)
        remote_name = 'scept_r1_split.json'
        files = set(self.api.list_repo_files(self.repo_id, repo_type='model', token=self.token))
        if self._path(remote_name) in files:
            if self._remote_bytes(remote_name) != value:
                raise ValueError('REMOTE frozen patient split differs: refusing training')
            return 'already_pinned'
        self._upload(remote_name, value)
        if self._remote_bytes(remote_name) != value:
            raise OSError('Remote frozen split mismatch')
        return 'uploaded_verified'

    def publish_latest(self, store):
        """Upload one verified generation; checkpoint BEFORE remote manifest."""
        candidates = store.verified()
        if not candidates:
            return None
        generation, slot, source, manifest = candidates[0]
        name = store.prefix
        key = (name, slot)
        if self._published.get(key) == (generation, manifest['sha256']):
            return manifest
        # Fail on mismatched contents before contacting the remote store.
        if digest_file(source) != manifest['sha256']:
            raise OSError('Local checkpoint hash differs from verified manifest')
        file_name = f'{name}_slot{slot}.pt'
        self._upload(file_name, source)
        # Hash a fresh remote download by streaming; don't hold GPU checkpoints
        # entirely in Kaggle RAM. This is independent of API upload success.
        with tempfile.TemporaryDirectory(prefix='scept-hf-check-') as tmp:
            downloaded = self._remote_file(file_name, tmp)
            if digest_file(downloaded) != manifest['sha256']:
                raise OSError('Remote checkpoint SHA256 mismatch; previous slot may still be valid')
        manifest_name = f'{name}_slot{slot}.json'
        raw = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
        self._upload(manifest_name, raw)
        if self._remote_bytes(manifest_name) != raw:
            raise OSError('Remote checkpoint manifest verification failed')
        self._published[key] = (generation, manifest['sha256'])
        return manifest

    def restore(self, store):
        """Fetch only checksum-verifiable remote generations into backup stage.

        No valid remote generation -> hard failure for --resume; cannot silently
        start a fresh model after a claimed recovery.
        """
        files = set(self.api.list_repo_files(self.repo_id, repo_type='model', token=self.token))
        recovered = []
        for slot in (0, 1):
            chk_name = f'{store.prefix}_slot{slot}.pt'
            man_name = f'{store.prefix}_slot{slot}.json'
            if self._path(chk_name) not in files or self._path(man_name) not in files:
                continue
            try:
                raw = self._remote_bytes(man_name)
                meta = json.loads(raw)
                if meta.get('slot') != slot or meta.get('prefix') != store.prefix:
                    continue
                if not isinstance(meta.get('generation'), int) or meta['generation'] < 0:
                    continue
                with tempfile.TemporaryDirectory(prefix='scept-hf-restore-') as tmp:
                    downloaded = self._remote_file(chk_name, tmp)
                    if downloaded.stat().st_size != meta['bytes'] or digest_file(downloaded) != meta['sha256']:
                        continue
                    existing = [m for m in store.verified() if m[1] == slot]
                    if not existing or existing[0][0] <= meta['generation']:
                        p, manifest_path = store._paths(store.backup, slot)
                        _atomic_bytes(p, downloaded)
                        _atomic_bytes(manifest_path, raw)
                    recovered.append(meta['generation'])
            except (OSError, ValueError, KeyError, TypeError):
                continue
        verified = store.verified()
        if not verified:
            raise FileNotFoundError(f'No SHA-verified remote generation for {store.prefix}')
        return {'highest_generation': verified[0][0], 'restored_generations': recovered}
