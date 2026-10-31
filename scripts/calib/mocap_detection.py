#!/usr/bin/env python3
"""Compatibility entry point for the ROS 2 mocap viewer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ros2/src/trimorph_mocap"))
from trimorph_mocap import mocap_detection

if __name__ == "__main__":
    raise SystemExit(mocap_detection.cli())
else:
    sys.modules[__name__] = mocap_detection
