"""A6 release qualification: guard documented API and honest version contract."""
from pathlib import Path
import importlib.metadata as metadata
import subprocess
import sys
import tomllib

import scept

ROOT = Path(__file__).resolve().parents[1]

def test_distribution_import_and_explicit_version():
    spec = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert spec["project"]["name"] == "scept-research"
    assert spec["project"]["version"] == scept.__version__
    assert metadata.version("scept-research") == scept.__version__
    assert spec["project"]["requires-python"] == ">=3.10"
    for entry in ("SCEPT", "CapturedRun", "load_observation", "ProtocolRegistry", "run_u10", "build_reference", "calibrate"):
        assert hasattr(scept, entry)

def test_independent_documented_example_executes():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "a6_clean_room.py")],
        text=True, capture_output=True, check=True,
    )
    assert '"status": "PASS_ENGINEERING_ONLY"' in result.stdout
    assert '"clinicalValidity": "NOT_ESTABLISHED"' in result.stdout
