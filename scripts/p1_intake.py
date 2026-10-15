#!/usr/bin/env python3
"""P1 CLI launcher; images and output stay outside Git."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainmirror.p1.intake import cli
if __name__ == "__main__":
    raise SystemExit(cli())
