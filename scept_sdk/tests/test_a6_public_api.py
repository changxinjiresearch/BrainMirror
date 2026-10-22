"""A6 release qualification: public API and version contract (Python 3.10+)."""
from pathlib import Path
import importlib.metadata as metadata
import re
import subprocess
import sys

import scept

ROOT = Path(__file__).resolve().parents[1]

def test_distribution_import_and_explicit_version():
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert re.search('^name = "scept-research"$', pyproject, flags=re.MULTILINE)
    version = re.search('^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    assert version is not None
    assert version.group(1) == scept.__version__
    assert metadata.version("scept-research") == scept.__version__
    assert re.search('^requires-python = ">=3.10"$', pyproject, flags=re.MULTILINE)
    for entry in ("SCEPT", "CapturedRun", "load_observation", "ProtocolRegistry", "run_u10", "build_reference", "calibrate"):
        assert hasattr(scept, entry)

def test_independent_documented_example_executes():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "a6_clean_room.py")],
        text=True, capture_output=True, check=True,
    )
    assert '"status": "PASS_ENGINEERING_ONLY"' in result.stdout
    assert '"clinicalValidity": "NOT_ESTABLISHED"' in result.stdout
