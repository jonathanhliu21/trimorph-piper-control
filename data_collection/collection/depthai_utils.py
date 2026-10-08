from __future__ import annotations

import time


_ANTI_BANDING_MODES = {
    "off": "OFF",
    "50": "MAINS_50_HZ",
    "60": "MAINS_60_HZ",
    "auto": "AUTO",
}


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


def apply_camera_control(dai, camera, *, brightness: int, anti_banding_hz: str) -> None:
    """Apply initial camera controls before starting the DepthAI pipeline."""

    if not -10 <= brightness <= 10:
        raise ValueError("Camera brightness must be in range -10..10")
    try:
        mode_name = _ANTI_BANDING_MODES[anti_banding_hz]
    except KeyError as exc:
        raise ValueError(
            "Camera anti-banding must be one of: off, 50, 60, auto"
        ) from exc
    camera.initialControl.setBrightness(brightness)
    mode = getattr(dai.CameraControl.AntiBandingMode, mode_name)
    camera.initialControl.setAntiBandingMode(mode)


def warmup_color_queue(queue, attempts: int) -> tuple[int, int] | None:
    """Read frames until one arrives, returning its (width, height)."""
    for _ in range(attempts):
        frame = queue.get().getCvFrame()
        if frame is not None:
            height, width = frame.shape[:2]
            return width, height
        time.sleep(0.05)
    return None
