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


def apply_camera_control(
    camera, *, exposure_us: int, iso: int, white_balance_k: int
) -> None:
    """Lock exposure and white balance before starting the DepthAI pipeline.

    Manual exposure disables auto-exposure (and with it brightness compensation
    and anti-banding); manual white balance disables auto white balance.
    """

    if exposure_us <= 0:
        raise ValueError("Camera exposure must be greater than 0 us")
    if not 100 <= iso <= 1600:
        raise ValueError("Camera ISO must be in range 100..1600")
    if not 1000 <= white_balance_k <= 12000:
        raise ValueError("Camera white balance must be in range 1000..12000 K")
    camera.initialControl.setManualExposure(exposure_us, iso)
    camera.initialControl.setManualWhiteBalance(white_balance_k)


def timedelta_ns(value) -> int:
    """Exact integer nanoseconds of a datetime.timedelta (microsecond resolution)."""

    return (value.days * 86_400 + value.seconds) * 1_000_000_000 + value.microseconds * 1_000


def host_clock_offset_ns(dai) -> int:
    """time.monotonic_ns() minus dai.Clock.now().

    Both use CLOCK_MONOTONIC on Linux, so this is ~0; measuring it keeps frame
    timestamps on the collection clock if the two ever differ.
    """

    before_ns = time.monotonic_ns()
    clock_ns = timedelta_ns(dai.Clock.now())
    after_ns = time.monotonic_ns()
    return (before_ns + after_ns) // 2 - clock_ns


def frame_capture_monotonic_ns(dai, frame, clock_offset_ns: int) -> int:
    """Mid-exposure capture time of a DepthAI frame on the host monotonic_ns clock.

    DepthAI converts the device timestamp to host time with its own continuous
    clock sync, so USB transfer and queueing delay are excluded.
    """

    stamp = frame.getTimestamp(dai.CameraExposureOffset.MIDDLE)
    return timedelta_ns(stamp) + clock_offset_ns


def warmup_color_queue(queue, attempts: int) -> tuple[int, int] | None:
    """Read frames until one arrives, returning its (width, height)."""
    for _ in range(attempts):
        frame = queue.get().getCvFrame()
        if frame is not None:
            height, width = frame.shape[:2]
            return width, height
        time.sleep(0.05)
    return None
