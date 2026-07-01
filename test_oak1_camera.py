"""Quick smoke test for an OAK-1 camera over depthai.

If a display is available (DISPLAY env var set), shows a live preview window
(press 'q' or Esc to quit). Otherwise runs headless: captures frames for a
few seconds and writes a snapshot PNG so you can confirm the feed looks right.

Usage:
    .venv/bin/python test_oak1_camera.py
"""

from __future__ import annotations

import os
import time

import cv2
import depthai as dai

RESOLUTION = (1280, 720)
HEADLESS_DURATION_S = 5.0
SNAPSHOT_PATH = "oak1_snapshot.png"


def main() -> None:
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        raise RuntimeError(
            "No depthai devices found. Check the USB connection and udev rules "
            "(/etc/udev/rules.d/80-movidius.rules), then unplug/replug the camera."
        )
    print(f"Found device(s): {[d.name for d in devices]}")

    headless = not os.environ.get("DISPLAY")
    if headless:
        print("No DISPLAY detected - running headless, will save a snapshot instead.")

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput(RESOLUTION, dai.ImgFrame.Type.NV12)
        queue = cam_out.createOutputQueue()

        pipeline.start()

        frame_count = 0
        last_frame = None
        start = time.monotonic()

        if headless:
            print(f"Capturing for {HEADLESS_DURATION_S:.0f}s...")
            while pipeline.isRunning() and (time.monotonic() - start) < HEADLESS_DURATION_S:
                last_frame = queue.get().getCvFrame()
                frame_count += 1
        else:
            print("Streaming... press 'q' or Esc to quit.")
            while pipeline.isRunning():
                last_frame = queue.get().getCvFrame()
                frame_count += 1

                elapsed = time.monotonic() - start
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                cv2.putText(
                    last_frame,
                    f"{last_frame.shape[1]}x{last_frame.shape[0]} @ {fps:.1f} fps",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("OAK-1 preview", last_frame)

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            cv2.destroyAllWindows()

        elapsed = time.monotonic() - start
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        print(f"Done. Captured {frame_count} frames in {elapsed:.1f}s ({fps:.1f} fps avg).")

        if headless and last_frame is not None:
            cv2.imwrite(SNAPSHOT_PATH, last_frame)
            print(f"Saved snapshot to {os.path.abspath(SNAPSHOT_PATH)}")


if __name__ == "__main__":
    main()
