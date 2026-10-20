#!/usr/bin/env python3
"""Display a live feed from the DepthAI wrist camera.

Run from the repository root:

    python3 scripts/camera/test_wrist_camera.py

Press ``q`` or Escape in the preview window to quit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Direct script execution only adds scripts/camera to sys.path. Add the
# repository root so this test uses the same helpers as data collection.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_collection.collection import config as defaults
from data_collection.collection.depthai_task_camera_worker import flip_frame
from data_collection.collection.depthai_utils import (
    apply_camera_control,
    import_depthai,
    require_depthai_device,
    warmup_color_queue,
)
from data_collection.collection.utils import import_cv2


WARMUP_ATTEMPTS = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a live feed from the DepthAI/OAK wrist camera."
    )
    parser.add_argument("--width", type=int, default=defaults.TASK_CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=defaults.TASK_CAMERA_HEIGHT)
    flip_group = parser.add_mutually_exclusive_group()
    flip_group.add_argument(
        "--flip",
        choices=("none", "horizontal", "vertical", "both", "180"),
        default=defaults.TASK_CAMERA_FLIP,
    )
    flip_group.add_argument(
        "--flip-180",
        action="store_true",
        help="rotate the camera image 180 degrees",
    )
    parser.add_argument("--window-name", default="wrist_camera")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be greater than zero")
    return args


def main() -> int:
    args = parse_args()
    cv2 = import_cv2()
    dai = import_depthai()
    require_depthai_device(dai)
    flip_mode = "both" if args.flip_180 else args.flip

    try:
        with dai.Pipeline() as pipeline:
            camera = pipeline.create(dai.node.Camera).build()
            apply_camera_control(
                dai,
                camera,
                brightness=defaults.TASK_CAMERA_BRIGHTNESS,
                anti_banding_hz=defaults.TASK_CAMERA_ANTI_BANDING_HZ,
            )
            camera_output = camera.requestOutput(
                (args.width, args.height), dai.ImgFrame.Type.NV12
            )
            queue = camera_output.createOutputQueue()
            pipeline.start()

            size = warmup_color_queue(queue, WARMUP_ATTEMPTS)
            if size is None:
                raise RuntimeError("Wrist camera produced no frames during warmup")

            actual_width, actual_height = size
            print(
                f"Wrist camera (DepthAI): {actual_width}x{actual_height}. "
                "Press q or Escape to quit."
            )

            while pipeline.isRunning():
                frame = queue.get().getCvFrame()
                if frame is None:
                    continue
                frame = flip_frame(cv2, frame, flip_mode)
                cv2.imshow(args.window_name, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
