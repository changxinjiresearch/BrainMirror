from pathlib import Path
import importlib.util
from types import SimpleNamespace
import os
import pytest

FILE=Path(__file__).resolve().parents[1]/'scripts'/'ixi_run_all.py'
spec=importlib.util.spec_from_file_location('ixi_run_all',FILE)
app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app)

def test_require_backup_without_repo_fails(monkeypatch):
    monkeypatch.delenv('BRAINMIRROR_BACKUP_TOKEN',raising=False)
    cfg=SimpleNamespace(backup_repo='',require_backup=True,backup_script='/tmp/nonexistent')
    with pytest.raises(RuntimeError,match='Backup is required'):
        app.backup_checkpoint(cfg)


def test_configured_backup_without_token_refuses(monkeypatch):
    monkeypatch.delenv('BRAINMIRROR_BACKUP_TOKEN',raising=False)
    cfg=SimpleNamespace(backup_repo='owner/private',require_backup=True,backup_script='/tmp/nonexistent')
    with pytest.raises(RuntimeError,match='Missing BRAINMIRROR_BACKUP_TOKEN'):
        app.backup_checkpoint(cfg)