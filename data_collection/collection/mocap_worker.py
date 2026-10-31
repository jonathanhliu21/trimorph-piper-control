"""Compatibility import for the ROS 2 mocap package in this repository."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ros2/src/trimorph_mocap"))
from trimorph_mocap import mocap_worker

# Preserve module identity for existing callers and mocks.
sys.modules[__name__] = mocap_worker
