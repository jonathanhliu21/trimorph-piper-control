"""Raw data-collection package for Midas/Piper demonstrations."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TELEOP_REPO = Path("/home/alvinz/alvinz/midas_hand_teleop")

for path in (REPO_ROOT, TELEOP_REPO):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
