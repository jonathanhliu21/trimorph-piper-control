from __future__ import annotations

import time


def import_depthai():
    try:
        import depthai as dai
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Luxonis depthai package. Install it with `pip install depthai` "
            "to use a depthai (OAK) camera."
        ) from exc
    return dai


def require_depthai_device(dai) -> None:
    if not dai.Device.getAllAvailableDevices():
        raise RuntimeError(
            "No depthai devices found. Check the USB connection and udev rules "
            "(/etc/udev/rules.d/80-movidius.rules), then unplug/replug the camera."
        )


def warmup_color_queue(queue, attempts: int) -> tuple[int, int] | None:
    """Read frames until one arrives, returning its (width, height)."""
    for _ in range(attempts):
        frame = queue.get().getCvFrame()
        if frame is not None:
            height, width = frame.shape[:2]
            return width, height
        time.sleep(0.05)
    return None
