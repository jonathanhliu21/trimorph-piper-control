from __future__ import annotations

import numpy as np

from .config import CollectionConfig
from .episode import RecordingManager
from .shared_state import SharedState
from .utils import import_cv2


def import_realsense():
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Intel RealSense Python bindings. Install librealsense/pyrealsense2."
        ) from exc
    return rs


def flip_frame(cv2, frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "horizontal":
        return cv2.flip(frame, 1)
    if mode == "vertical":
        return cv2.flip(frame, 0)
    if mode == "both":
        return cv2.flip(frame, -1)
    return frame


def device_info(rs, device) -> dict[str, str]:
    def get_info(info) -> str:
        if not device.supports(info):
            return ""
        return str(device.get_info(info))

    return {
        "name": get_info(rs.camera_info.name) or "Unknown RealSense",
        "serial": get_info(rs.camera_info.serial_number) or "unknown-serial",
        "firmware": get_info(rs.camera_info.firmware_version) or "unknown-firmware",
        "usb": get_info(rs.camera_info.usb_type_descriptor) or "unknown-usb",
    }


def find_realsense_device(rs, serial: str | None):
    devices = rs.context().query_devices()
    for device in devices:
        info = device_info(rs, device)
        if serial is None or serial == info["serial"]:
            return device
    if serial:
        raise RuntimeError(f"No RealSense device found with serial {serial}")
    raise RuntimeError("No RealSense devices found")


def color_profiles(rs, device) -> list[dict[str, int]]:
    profiles: list[dict[str, int]] = []
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != rs.stream.color:
                continue
            video_profile = profile.as_video_stream_profile()
            if video_profile.format() != rs.format.rgb8:
                continue
            profiles.append(
                {
                    "width": int(video_profile.width()),
                    "height": int(video_profile.height()),
                    "fps": int(video_profile.fps()),
                }
            )
    return sorted(profiles, key=lambda item: (item["width"], item["height"], item["fps"]))


def choose_color_profile(rs, device, config: CollectionConfig) -> dict[str, int]:
    profiles = color_profiles(rs, device)
    for profile in profiles:
        if (
            profile["width"] == config.realsense_width
            and profile["height"] == config.realsense_height
            and profile["fps"] == config.realsense_fps
        ):
            return profile
    supported = ", ".join(f"{p['width']}x{p['height']}@{p['fps']}" for p in profiles)
    raise RuntimeError(
        f"No RealSense RGB8 profile for {config.realsense_width}x{config.realsense_height}@"
        f"{config.realsense_fps}. Supported: {supported or '(none)'}"
    )


def format_realsense_devices() -> str:
    rs = import_realsense()
    devices = rs.context().query_devices()
    if len(devices) == 0:
        return "No RealSense devices found."
    lines = []
    for index, device in enumerate(devices):
        info = device_info(rs, device)
        lines.append(f"[{index}] {info['name']}")
        lines.append(f"  serial: {info['serial']}")
        lines.append(f"  firmware: {info['firmware']}")
        lines.append(f"  usb: {info['usb']}")
        profiles = color_profiles(rs, device)
        if profiles:
            compact = ", ".join(
                f"{p['width']}x{p['height']}@{p['fps']}" for p in profiles
            )
            lines.append(f"  rgb8 profiles: {compact}")
    return "\n".join(lines)


def run_realsense_worker(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    *,
    init_done,
) -> None:
    """Supplies RealSense color/depth to teleop wrist-pose estimation.

    Never recorded to video — RealSense isn't a task-camera source anymore,
    so `recorder` is accepted only to match the generic worker signature.
    """
    cv2 = import_cv2()
    rs = import_realsense()
    pipeline = rs.pipeline()
    align = rs.align(rs.stream.color)
    timeout_count = 0

    device = find_realsense_device(rs, config.realsense_serial)
    selected = choose_color_profile(rs, device, config)
    info = device_info(rs, device)
    if info["usb"].startswith("2"):
        print("Warning: RealSense reports USB 2.x; use USB 3 for reliable 640x480@30.")

    rs_config = rs.config()
    rs_config.enable_device(info["serial"])
    rs_config.enable_stream(
        rs.stream.color,
        selected["width"],
        selected["height"],
        rs.format.rgb8,
        selected["fps"],
    )
    rs_config.enable_stream(
        rs.stream.depth,
        selected["width"],
        selected["height"],
        rs.format.z16,
        selected["fps"],
    )

    pipeline.start(rs_config)
    print(
        "RealSense stream (teleop depth only): "
        f"{selected['width']}x{selected['height']} rgb8 @{selected['fps']}fps "
        f"serial={info['serial']}"
    )
    warmup_timeout_ms = max(config.realsense_timeout_ms, 3000)
    warmup_received = 0
    warmup_attempts = 0
    warmup_max_attempts = config.realsense_warmup_frames * 6
    while warmup_received < config.realsense_warmup_frames and warmup_attempts < warmup_max_attempts:
        warmup_attempts += 1
        try:
            pipeline.wait_for_frames(warmup_timeout_ms)
            warmup_received += 1
        except RuntimeError as exc:
            if "Frame didn't arrive" not in str(exc):
                raise
    if warmup_received == 0:
        raise RuntimeError("RealSense produced no frames during warmup")

    init_done.set()
    try:
        while not shared.shutdown.is_set():
            try:
                frames = pipeline.wait_for_frames(config.realsense_timeout_ms)
            except RuntimeError as exc:
                if "Frame didn't arrive" not in str(exc):
                    raise
                timeout_count += 1
                if config.realsense_max_timeouts and timeout_count >= config.realsense_max_timeouts:
                    raise RuntimeError(
                        f"No RealSense frames after {timeout_count} timeouts"
                    ) from exc
                continue

            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            timeout_count = 0
            rgb = np.asanyarray(color_frame.get_data())
            rgb = flip_frame(cv2, rgb, config.realsense_flip)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            shared.set_realsense_frame(bgr)
            try:
                intr = color_frame.profile.as_video_stream_profile().intrinsics
                intrinsics = {
                    "width": float(intr.width),
                    "height": float(intr.height),
                    "fx": float(intr.fx),
                    "fy": float(intr.fy),
                    "ppx": float(intr.ppx),
                    "ppy": float(intr.ppy),
                }
            except Exception:
                intrinsics = None

            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * float(depth_frame.get_units())
            depth_m = flip_frame(cv2, depth_m, config.realsense_flip)
            shared.set_realsense_depth(depth_m, intrinsics)
    finally:
        pipeline.stop()
