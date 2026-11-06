#!/usr/bin/env python3
"""Large live Piper-base XYZ display; never commands motion or changes settings.

Run from the repository root::

    .venv/bin/python scripts/piper_api/read_piper_visualizer.py --fullscreen

Position is the firmware flange origin, not a fingertip. XYZ defaults to mm;
RPY and joints use radians, matching read_piper.py. Space captures a reference
for displayed XYZ changes only. Q/Esc closes the viewer.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_collection.collection.piper_worker import piper_workspace_warning
from data_collection.collection.tracking_monitor import AXIS_COLORS
from data_collection.collection.utils import import_cv2
from scripts.piper_api.piper_api import PiperAPI

WINDOW = "Piper XYZ - read only"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--firmware", choices=("default", "v183", "v188"), default="v183")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--units", choices=("mm", "m"), default="mm")
    parser.add_argument("--rate-hz", type=float, default=10.)
    parser.add_argument("--stale-s", type=float, default=.5,
                        help="Hide feedback older than this many seconds (default: 0.5)")
    args = parser.parse_args(argv)
    for name in ("rate_hz", "stale_s"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    return args


def fresh_values(state, field, count, now, stale_s):
    """Never show missing/expired feedback as a current position."""
    if state is None:
        return None
    values = np.asarray(getattr(state, field), dtype=float)
    age = now - float(state.timestamp)
    if values.shape != (count,) or not np.isfinite(values).all() or not 0 <= age <= stale_s:
        return None
    return values


def render(cv2, pose, joints, reference, units, status, warning):
    image = np.full((900, 1280, 3), 28, dtype=np.uint8)
    def text(value, x, y, scale=.7, color=(235, 235, 235), thickness=1):
        cv2.putText(image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)
    def fit_text(value, x, y, max_width, scale, color, thickness):
        width = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        text(value, x, y, min(scale, scale * max_width / max(width, 1)), color, thickness)
    text("PIPER BASE XYZ", 24, 40, 1., thickness=2)
    text(f"Flange origin | {units} | read only", 720, 38)
    text(status, 24, 78, color=(80, 200, 80) if pose is not None else (40, 170, 255))
    multiplier, precision = (1000., 1) if units == "mm" else (1., 4)
    for axis, (name, color) in enumerate(zip("XYZ", AXIS_COLORS)):
        top = 105 + axis * 205
        cv2.rectangle(image, (20, top), (1260, top + 190), (40, 40, 40), -1)
        text(name, 45, top + 135, 4.5, color, 9)
        value = "---" if pose is None else f"{pose[axis] * multiplier:+.{precision}f}"
        fit_text(value, 220, top + 135, 770, 4.8, color, 8)
        text(units, 1110, top + 130, 1.4, color, 2)
        if pose is not None and reference is not None:
            change = (pose[axis] - reference[axis]) * multiplier
            text(f"Change from reference: {change:+.{precision}f} {units}", 230, top + 175, .65)
    if pose is not None:
        text("RPY [rad]: " + "   ".join(f"{name} {value:+.4f}" for name, value in zip(("R", "P", "Y"), pose[3:])),
             24, 755)
    if joints is not None:
        text("Joints [rad]: " + "   ".join(f"J{i+1} {value:+.3f}" for i, value in enumerate(joints)), 24, 790, .6)
    if warning:
        fit_text(warning, 24, 830, 1230, .6, (70, 100, 255), 1)
    text("Space: set display reference    F: fullscreen    Q / Esc: quit", 24, 880, .65)
    return image


def run(args):
    cv2 = import_cv2()
    # None is essential: PiperAPI.connect normally applies installation/payload
    # settings. With these unset, it only connects and starts feedback reception.
    api = PiperAPI(firmware=args.firmware, interface=args.interface, channel=args.channel,
                   installation_pos=None, payload=None)
    reference = None
    fullscreen = args.fullscreen
    window_open = False
    try:
        if not api.connect():
            raise RuntimeError(f"Piper is not responding on {args.channel}")
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        window_open = True
        cv2.resizeWindow(WINDOW, 1280, 900)
        if fullscreen:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while True:
            start = time.monotonic()
            eef_state, joint_state = api.get_eef_state(), api.get_joint_state()
            now = time.time()
            pose = fresh_values(eef_state, "pose", 6, now, args.stale_s)
            joints = fresh_values(joint_state, "joints", 6, now, args.stale_s)
            status = (f"Live flange feedback on {args.channel}" if pose is not None
                      else "NO FRESH FLANGE FEEDBACK - values hidden")
            warning = piper_workspace_warning(api.get_state())
            cv2.imshow(WINDOW, render(cv2, pose, joints, reference, args.units, status, warning))
            key = cv2.waitKey(max(1, int((1 / args.rate_hz - (time.monotonic() - start)) * 1000))) & 0xff
            if key in (27, ord('q')) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(' ') and pose is not None:
                reference = pose[:3].copy()
            elif key == ord('f'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    finally:
        api.disconnect()
        if window_open:
            cv2.destroyWindow(WINDOW)
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
