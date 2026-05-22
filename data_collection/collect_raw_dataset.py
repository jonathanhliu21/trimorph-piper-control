#!/usr/bin/env python3
"""Collect synchronized raw RealSense, Midas hand, and Piper EEF episodes."""

from __future__ import annotations

import csv
import json
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.midas_api import DEFAULT_CONFIG_PATH, connect_hand  # noqa: E402
from scripts.piper_api.foot_pedal import (  # noqa: E402
    FootPedalTrigger,
    format_input_devices,
    list_input_devices,
)
from scripts.piper_api.teacher_mode import get_teacher_mode  # noqa: E402


# Operator settings. Adjust these values, then run:
#   python3 data_collection/collect_raw_dataset.py
DATASET_ROOT = Path("data_midas")
LIST_INPUT_DEVICES = False

READY_SECONDS = 1.0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_SERIAL: str | None = None
CAMERA_FLIP = "none"  # one of: "none", "horizontal", "vertical", "both"
CAMERA_TIMEOUT_MS = 500
CAMERA_MAX_TIMEOUTS = 30
CAMERA_WARMUP_FRAMES = 30
VIDEO_CODEC = "mp4v"

MIDAS_RATE_HZ = 200.0
ENABLE_HAND_TORQUE = False
HAND_PORT: str | None = None
HAND_BAUDRATE = 1_000_000
HAND_CONFIG = str(DEFAULT_CONFIG_PATH)

PIPER_RATE_HZ = 200.0
PIPER_INTERFACE = "socketcan"
PIPER_CHANNEL = "can0"
PIPER_FIRMWARE = "default"  # one of: "default", "v183", "v188"
PIPER_INSTALLATION_POS: str | None = "right"
TEACHER_MODE_POLL_SECONDS = 0.25

FOOT_PEDAL_DEVICE: str | None = None
FOOT_PEDAL_KEY_CODE: int | None = None
FOOT_PEDAL_DEBOUNCE_SECONDS = 0.5


CANONICAL_MIDAS_MOTOR_IDS = tuple(range(13))
RAW_EEF_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class CollectionConfig:
    dataset_root: Path
    list_input_devices: bool
    ready_seconds: float
    camera_width: int
    camera_height: int
    camera_fps: int
    camera_serial: str | None
    camera_flip: str
    camera_timeout_ms: int
    camera_max_timeouts: int
    camera_warmup_frames: int
    video_codec: str
    midas_rate_hz: float
    enable_hand_torque: bool
    port: str | None
    baudrate: int
    config: str
    piper_rate_hz: float
    interface: str
    channel: str
    firmware: str
    installation_pos: str | None
    teacher_mode_poll_seconds: float
    foot_pedal_device: str | None
    foot_pedal_key_code: int | None
    foot_pedal_debounce_seconds: float


@dataclass(frozen=True)
class EpisodePaths:
    episode_index: int
    episode_dir: Path
    video_path: Path
    camera_timestamps_path: Path
    midas_path: Path
    piper_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class WorkerError:
    name: str
    message: str
    traceback_text: str


def build_config() -> CollectionConfig:
    return CollectionConfig(
        dataset_root=Path(DATASET_ROOT).expanduser(),
        list_input_devices=LIST_INPUT_DEVICES,
        ready_seconds=READY_SECONDS,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        camera_fps=CAMERA_FPS,
        camera_serial=CAMERA_SERIAL,
        camera_flip=CAMERA_FLIP,
        camera_timeout_ms=CAMERA_TIMEOUT_MS,
        camera_max_timeouts=CAMERA_MAX_TIMEOUTS,
        camera_warmup_frames=CAMERA_WARMUP_FRAMES,
        video_codec=VIDEO_CODEC,
        midas_rate_hz=MIDAS_RATE_HZ,
        enable_hand_torque=ENABLE_HAND_TORQUE,
        port=HAND_PORT,
        baudrate=HAND_BAUDRATE,
        config=str(Path(HAND_CONFIG).expanduser()),
        piper_rate_hz=PIPER_RATE_HZ,
        interface=PIPER_INTERFACE,
        channel=PIPER_CHANNEL,
        firmware=PIPER_FIRMWARE,
        installation_pos=PIPER_INSTALLATION_POS,
        teacher_mode_poll_seconds=TEACHER_MODE_POLL_SECONDS,
        foot_pedal_device=FOOT_PEDAL_DEVICE,
        foot_pedal_key_code=FOOT_PEDAL_KEY_CODE,
        foot_pedal_debounce_seconds=FOOT_PEDAL_DEBOUNCE_SECONDS,
    )


def validate_config(config: CollectionConfig) -> None:
    positive_float_fields = {
        "READY_SECONDS": config.ready_seconds,
        "MIDAS_RATE_HZ": config.midas_rate_hz,
        "PIPER_RATE_HZ": config.piper_rate_hz,
        "TEACHER_MODE_POLL_SECONDS": config.teacher_mode_poll_seconds,
    }
    for name, value in positive_float_fields.items():
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0")
    if config.camera_width <= 0:
        raise ValueError("CAMERA_WIDTH must be greater than 0")
    if config.camera_height <= 0:
        raise ValueError("CAMERA_HEIGHT must be greater than 0")
    if config.camera_fps <= 0:
        raise ValueError("CAMERA_FPS must be greater than 0")
    if config.camera_flip not in {"none", "horizontal", "vertical", "both"}:
        raise ValueError("CAMERA_FLIP must be one of: none, horizontal, vertical, both")
    if config.camera_timeout_ms <= 0:
        raise ValueError("CAMERA_TIMEOUT_MS must be greater than 0")
    if config.camera_max_timeouts < 0:
        raise ValueError("CAMERA_MAX_TIMEOUTS must be 0 or greater")
    if config.camera_warmup_frames < 0:
        raise ValueError("CAMERA_WARMUP_FRAMES must be 0 or greater")
    if len(config.video_codec) != 4:
        raise ValueError("VIDEO_CODEC must be a 4-character fourcc")
    if config.firmware not in {"default", "v183", "v188"}:
        raise ValueError("PIPER_FIRMWARE must be one of: default, v183, v188")
    if config.foot_pedal_debounce_seconds < 0:
        raise ValueError("FOOT_PEDAL_DEBOUNCE_SECONDS must be 0 or greater")


def import_camera_dependencies():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing OpenCV Python bindings. Install with: python3 -m pip install opencv-python"
        ) from exc
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing NumPy. Install with: python3 -m pip install numpy") from exc
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Intel RealSense Python bindings. Install librealsense/pyrealsense2 "
            "for this Python environment."
        ) from exc
    return cv2, np, rs


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
    context = rs.context()
    devices = context.query_devices()
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
    profiles.sort(key=lambda item: (item["width"], item["height"], item["fps"]))
    return profiles


def choose_color_profile(rs, device, config: CollectionConfig) -> dict[str, int]:
    matches = [
        profile
        for profile in color_profiles(rs, device)
        if profile["width"] == config.camera_width
        and profile["height"] == config.camera_height
        and profile["fps"] == config.camera_fps
    ]
    if not matches:
        supported = ", ".join(
            f"{p['width']}x{p['height']}@{p['fps']}" for p in color_profiles(rs, device)
        )
        raise RuntimeError(
            "No matching RealSense color profile found for "
            f"{config.camera_width}x{config.camera_height}@{config.camera_fps}. "
            f"Supported RGB8 profiles: {supported or '(none)'}"
        )
    return matches[0]


def flip_frame(cv2, frame, mode: str):
    if mode == "horizontal":
        return cv2.flip(frame, 1)
    if mode == "vertical":
        return cv2.flip(frame, 0)
    if mode == "both":
        return cv2.flip(frame, -1)
    return frame


class RealSenseStream:
    def __init__(self, config: CollectionConfig):
        self.config = config
        self.cv2, self.np, self.rs = import_camera_dependencies()
        self.pipeline = self.rs.pipeline()
        self.metadata: dict[str, Any] = {}
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        device = find_realsense_device(self.rs, self.config.camera_serial)
        info = device_info(self.rs, device)
        selected = choose_color_profile(self.rs, device, self.config)

        if info["usb"].startswith("2"):
            print(
                "Warning: camera reports USB 2.x. RealSense 640x480@30 is more "
                "reliable on a USB 3 port/cable."
            )

        config = self.rs.config()
        config.enable_device(info["serial"])
        config.enable_stream(
            self.rs.stream.color,
            selected["width"],
            selected["height"],
            self.rs.format.rgb8,
            selected["fps"],
        )
        active_profile = self.pipeline.start(config)
        active_info = device_info(self.rs, active_profile.get_device())
        self.metadata = {
            "device": active_info,
            "profile": selected,
            "flip": self.config.camera_flip,
            "timeout_ms": self.config.camera_timeout_ms,
            "video_codec": self.config.video_codec,
        }
        self.started = True
        print(
            "RealSense stream: "
            f"{selected['width']}x{selected['height']} rgb8 @{selected['fps']}fps "
            f"serial={active_info['serial']}"
        )
        for _ in range(self.config.camera_warmup_frames):
            self.pipeline.wait_for_frames(self.config.camera_timeout_ms)

    def stop(self) -> None:
        if not self.started:
            return
        self.pipeline.stop()
        self.started = False

    def drain_pending_frames(self) -> int:
        drained = 0
        while True:
            frames = self.pipeline.poll_for_frames()
            try:
                has_frames = bool(frames) and frames.size() > 0
            except AttributeError:
                has_frames = bool(frames)
            if not has_frames:
                return drained
            if frames.get_color_frame():
                drained += 1

    def read_color_frame(self):
        frames = self.pipeline.wait_for_frames(self.config.camera_timeout_ms)
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        rgb_image = self.np.asanyarray(color_frame.get_data())
        rgb_image = flip_frame(self.cv2, rgb_image, self.config.camera_flip)
        timestamp_domain = ""
        try:
            timestamp_domain = str(color_frame.get_frame_timestamp_domain())
        except Exception:
            timestamp_domain = ""
        return {
            "rgb_image": rgb_image,
            "realsense_timestamp_ms": float(color_frame.get_timestamp()),
            "realsense_frame_number": int(color_frame.get_frame_number()),
            "realsense_timestamp_domain": timestamp_domain,
        }


def utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def raw_root(dataset_root: Path) -> Path:
    return dataset_root / "raw"


def next_episode_index(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in root.glob("episode_*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("episode_")
        if suffix.isdigit():
            indices.append(int(suffix))
    return max(indices, default=-1) + 1


def episode_paths(dataset_root: Path, episode_index: int) -> EpisodePaths:
    episode_dir = raw_root(dataset_root) / f"episode_{episode_index:06d}"
    return EpisodePaths(
        episode_index=episode_index,
        episode_dir=episode_dir,
        video_path=episode_dir / "camera.mp4",
        camera_timestamps_path=episode_dir / "camera_timestamps.csv",
        midas_path=episode_dir / "midas.csv",
        piper_path=episode_dir / "piper.csv",
        metadata_path=episode_dir / "metadata.json",
    )


def wait_for_teacher_mode(api, *, poll_s: float) -> None:
    print("Waiting for Piper teacher mode before accepting a start pedal press.")
    next_notice_t = 0.0
    while True:
        is_teacher_mode, status = get_teacher_mode(api)
        if is_teacher_mode:
            print("Piper teacher mode is active.")
            return
        now = time.monotonic()
        if now >= next_notice_t:
            print(
                "Piper teacher mode is not active. "
                f"Current teach_status={status!r}. "
                "Put the robot in teacher mode to collect an episode."
            )
            next_notice_t = now + 2.0
        time.sleep(poll_s)


def wait_for_foot_pedal(pedal: FootPedalTrigger, message: str, *, poll_s: float = 0.01):
    print(message)
    while True:
        events = pedal.poll()
        if events:
            print(events[-1].message)
            return events[-1]
        time.sleep(poll_s)


def wait_ready_delay(config: CollectionConfig, api) -> bool:
    deadline = time.monotonic() + config.ready_seconds
    print(f"Recording starts in {config.ready_seconds:.1f}s. Keep Piper in teacher mode.")
    while True:
        now = time.monotonic()
        if now >= deadline:
            return True
        is_teacher_mode, status = get_teacher_mode(api)
        if not is_teacher_mode:
            print(
                "Start cancelled because Piper left teacher mode during the delay "
                f"(teach_status={status!r})."
            )
            return False
        time.sleep(min(0.05, deadline - now))


def sleep_until_ns(target_ns: int, stop_event: threading.Event) -> bool:
    while not stop_event.is_set():
        now_ns = time.monotonic_ns()
        remaining_s = (target_ns - now_ns) / 1_000_000_000
        if remaining_s <= 0:
            return True
        stop_event.wait(min(remaining_s, 0.02))
    return False


def worker_wrapper(
    name: str,
    errors: queue.Queue[WorkerError],
    stop_event: threading.Event,
    target,
    *args,
) -> None:
    try:
        target(*args)
    except Exception as exc:
        errors.put(
            WorkerError(
                name=name,
                message=str(exc),
                traceback_text=traceback.format_exc(),
            )
        )
        stop_event.set()


def record_camera(
    stream: RealSenseStream,
    paths: EpisodePaths,
    config: CollectionConfig,
    stop_event: threading.Event,
    episode_start_ns: int,
    counts: dict[str, int],
) -> None:
    cv2 = stream.cv2
    fourcc = cv2.VideoWriter_fourcc(*config.video_codec)
    writer = cv2.VideoWriter(
        str(paths.video_path),
        fourcc,
        float(config.camera_fps),
        (config.camera_width, config.camera_height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open video writer for {paths.video_path} with codec {config.video_codec!r}"
        )

    fieldnames = [
        "frame_idx",
        "monotonic_ns",
        "wall_time_s",
        "elapsed_s",
        "realsense_timestamp_ms",
        "realsense_frame_number",
        "realsense_timestamp_domain",
    ]
    frame_idx = 0
    timeout_count = 0
    next_timeout_notice_t = 0.0
    try:
        with paths.camera_timestamps_path.open("w", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            while not stop_event.is_set():
                try:
                    frame = stream.read_color_frame()
                except RuntimeError as exc:
                    if "Frame didn't arrive" not in str(exc):
                        raise
                    timeout_count += 1
                    now = time.monotonic()
                    if now >= next_timeout_notice_t:
                        print("Still waiting for RealSense frames...")
                        next_timeout_notice_t = now + 2.0
                    if (
                        config.camera_max_timeouts
                        and timeout_count >= config.camera_max_timeouts
                    ):
                        raise RuntimeError(
                            f"No RealSense frames arrived after {timeout_count} timeouts"
                        ) from exc
                    continue
                if frame is None:
                    continue
                timeout_count = 0
                sample_ns = time.monotonic_ns()
                wall_time_s = time.time()
                writer.write(cv2.cvtColor(frame["rgb_image"], cv2.COLOR_RGB2BGR))
                csv_writer.writerow(
                    {
                        "frame_idx": frame_idx,
                        "monotonic_ns": sample_ns,
                        "wall_time_s": f"{wall_time_s:.6f}",
                        "elapsed_s": f"{(sample_ns - episode_start_ns) / 1e9:.9f}",
                        "realsense_timestamp_ms": f"{frame['realsense_timestamp_ms']:.6f}",
                        "realsense_frame_number": frame["realsense_frame_number"],
                        "realsense_timestamp_domain": frame[
                            "realsense_timestamp_domain"
                        ],
                    }
                )
                frame_idx += 1
                if frame_idx % max(1, config.camera_fps) == 0:
                    csv_file.flush()
    finally:
        writer.release()
        counts["camera_frames"] = frame_idx


def record_midas(
    hand,
    paths: EpisodePaths,
    config: CollectionConfig,
    stop_event: threading.Event,
    episode_start_ns: int,
    counts: dict[str, int],
) -> None:
    motor_ids = [int(motor_id) for motor_id in hand.motor_ids]
    if len(motor_ids) != len(CANONICAL_MIDAS_MOTOR_IDS):
        raise RuntimeError(
            f"Expected 13 Midas motors, got {len(motor_ids)} motor_ids={motor_ids}"
        )
    fieldnames = ["sample_idx", "monotonic_ns", "wall_time_s", "elapsed_s"]
    fieldnames.extend(f"pos_{motor_id}" for motor_id in motor_ids)
    fieldnames.extend(f"vel_{motor_id}" for motor_id in motor_ids)
    fieldnames.extend(f"cur_{motor_id}" for motor_id in motor_ids)

    period_ns = int(round(1_000_000_000 / config.midas_rate_hz))
    next_sample_ns = time.monotonic_ns()
    sample_idx = 0
    with paths.midas_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        while sleep_until_ns(next_sample_ns, stop_event):
            pos, vel, cur = hand.read_pos_vel_cur()
            sample_ns = time.monotonic_ns()
            row = {
                "sample_idx": sample_idx,
                "monotonic_ns": sample_ns,
                "wall_time_s": f"{time.time():.6f}",
                "elapsed_s": f"{(sample_ns - episode_start_ns) / 1e9:.9f}",
            }
            for motor_id, value in zip(motor_ids, pos):
                row[f"pos_{motor_id}"] = f"{float(value):.9f}"
            for motor_id, value in zip(motor_ids, vel):
                row[f"vel_{motor_id}"] = f"{float(value):.9f}"
            for motor_id, value in zip(motor_ids, cur):
                row[f"cur_{motor_id}"] = f"{float(value):.9f}"
            writer.writerow(row)
            sample_idx += 1
            if sample_idx % max(1, int(config.midas_rate_hz)) == 0:
                csv_file.flush()
            next_sample_ns += period_ns
            while next_sample_ns < sample_ns:
                next_sample_ns += period_ns
    counts["midas_samples"] = sample_idx


def record_piper(
    api,
    paths: EpisodePaths,
    config: CollectionConfig,
    stop_event: threading.Event,
    episode_start_ns: int,
    counts: dict[str, int],
) -> None:
    fieldnames = [
        "sample_idx",
        "monotonic_ns",
        "wall_time_s",
        "elapsed_s",
        "robot_timestamp_s",
        "robot_hz",
        *RAW_EEF_FIELDS,
    ]
    period_ns = int(round(1_000_000_000 / config.piper_rate_hz))
    next_sample_ns = time.monotonic_ns()
    sample_idx = 0
    with paths.piper_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        while sleep_until_ns(next_sample_ns, stop_event):
            eef = api.get_eef_state()
            sample_ns = time.monotonic_ns()
            next_sample_ns += period_ns
            while next_sample_ns < sample_ns:
                next_sample_ns += period_ns
            if eef is None:
                continue
            row = {
                "sample_idx": sample_idx,
                "monotonic_ns": sample_ns,
                "wall_time_s": f"{time.time():.6f}",
                "elapsed_s": f"{(sample_ns - episode_start_ns) / 1e9:.9f}",
                "robot_timestamp_s": f"{eef.timestamp:.6f}",
                "robot_hz": f"{eef.hz:.3f}",
            }
            row.update(
                {name: f"{float(value):.9f}" for name, value in zip(RAW_EEF_FIELDS, eef.pose)}
            )
            writer.writerow(row)
            sample_idx += 1
            if sample_idx % max(1, int(config.piper_rate_hz)) == 0:
                csv_file.flush()
    counts["piper_samples"] = sample_idx


def build_base_metadata(
    config: CollectionConfig,
    paths: EpisodePaths,
    camera_metadata: dict[str, Any],
    hand,
) -> dict[str, Any]:
    return {
        "episode_index": paths.episode_index,
        "status": "recording",
        "created_utc": utc_stamp(),
        "paths": {
            "camera_mp4": paths.video_path.name,
            "camera_timestamps_csv": paths.camera_timestamps_path.name,
            "midas_csv": paths.midas_path.name,
            "piper_csv": paths.piper_path.name,
        },
        "timing": {
            "sync_clock": "host monotonic_ns",
            "ready_seconds": config.ready_seconds,
        },
        "camera": {
            "target_width": config.camera_width,
            "target_height": config.camera_height,
            "target_fps": config.camera_fps,
            **camera_metadata,
        },
        "midas": {
            "target_rate_hz": config.midas_rate_hz,
            "motor_ids": [int(motor_id) for motor_id in hand.motor_ids],
            "canonical_motor_ids": list(CANONICAL_MIDAS_MOTOR_IDS),
            "position_units": "calibrated radians from midas_hand_api.read_pos_vel_cur()",
            "torque_enabled_by_collector": bool(config.enable_hand_torque),
            "port": getattr(hand, "port", None),
            "config": str(Path(config.config).expanduser()),
        },
        "piper": {
            "target_rate_hz": config.piper_rate_hz,
            "interface": config.interface,
            "channel": config.channel,
            "firmware": config.firmware,
            "installation_pos": config.installation_pos,
            "raw_eef_format": "[x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]",
            "processed_eef_format": "[x_m, y_m, z_m, rotvec_x_rad, rotvec_y_rad, rotvec_z_rad]",
            "euler_convention": "fixed-axis XYZ: R = Rz(yaw) * Ry(pitch) * Rx(roll)",
        },
        "foot_pedal": {
            "device": config.foot_pedal_device,
            "key_code": config.foot_pedal_key_code,
            "debounce_seconds": config.foot_pedal_debounce_seconds,
        },
    }


def record_episode(
    *,
    paths: EpisodePaths,
    config: CollectionConfig,
    camera: RealSenseStream,
    hand,
    api,
    pedal: FootPedalTrigger,
) -> None:
    paths.episode_dir.mkdir(parents=True, exist_ok=False)
    episode_start_ns = time.monotonic_ns()
    metadata = build_base_metadata(config, paths, camera.metadata, hand)
    metadata["started_utc"] = utc_stamp()
    metadata["started_monotonic_ns"] = episode_start_ns
    write_json(paths.metadata_path, metadata)

    stop_event = threading.Event()
    errors: queue.Queue[WorkerError] = queue.Queue()
    counts: dict[str, int] = {}

    workers = [
        threading.Thread(
            target=worker_wrapper,
            name="camera-recorder",
            args=(
                "camera",
                errors,
                stop_event,
                record_camera,
                camera,
                paths,
                config,
                stop_event,
                episode_start_ns,
                counts,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=worker_wrapper,
            name="midas-recorder",
            args=(
                "midas",
                errors,
                stop_event,
                record_midas,
                hand,
                paths,
                config,
                stop_event,
                episode_start_ns,
                counts,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=worker_wrapper,
            name="piper-recorder",
            args=(
                "piper",
                errors,
                stop_event,
                record_piper,
                api,
                paths,
                config,
                stop_event,
                episode_start_ns,
                counts,
            ),
            daemon=True,
        ),
    ]
    for worker in workers:
        worker.start()

    print(
        f"Recording episode {paths.episode_index:06d}. "
        "Press the foot pedal again to stop this episode."
    )
    status = "complete"
    error_records: list[dict[str, str]] = []
    try:
        while not stop_event.is_set():
            if pedal.poll():
                print("Foot pedal pressed; stopping episode.")
                stop_event.set()
                break
            try:
                error = errors.get_nowait()
            except queue.Empty:
                error = None
            if error is not None:
                status = "error"
                error_records.append(
                    {
                        "worker": error.name,
                        "message": error.message,
                        "traceback": error.traceback_text,
                    }
                )
                print(f"{error.name} recorder failed: {error.message}")
                stop_event.set()
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        status = "interrupted"
        print("\nStopping current episode before exiting.")
        stop_event.set()
        raise
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=5.0)
        while True:
            try:
                error = errors.get_nowait()
            except queue.Empty:
                break
            status = "error"
            error_records.append(
                {
                    "worker": error.name,
                    "message": error.message,
                    "traceback": error.traceback_text,
                }
            )
        episode_stop_ns = time.monotonic_ns()
        metadata.update(
            {
                "status": status,
                "stopped_utc": utc_stamp(),
                "stopped_monotonic_ns": episode_stop_ns,
                "duration_s": (episode_stop_ns - episode_start_ns) / 1e9,
                "counts": counts,
                "errors": error_records,
            }
        )
        write_json(paths.metadata_path, metadata)
        print(
            f"Saved episode {paths.episode_index:06d}: "
            f"{counts.get('camera_frames', 0)} frames, "
            f"{counts.get('midas_samples', 0)} Midas samples, "
            f"{counts.get('piper_samples', 0)} Piper samples."
        )
    if status == "error":
        raise RuntimeError(f"Episode {paths.episode_index:06d} failed; see metadata.json")


def main() -> int:
    config = build_config()
    if config.list_input_devices:
        print(format_input_devices(list_input_devices()))
        return 0
    validate_config(config)

    from scripts.piper_api.piper_api import PiperAPI

    dataset_root = config.dataset_root
    dataset_root.mkdir(parents=True, exist_ok=True)
    raw_root(dataset_root).mkdir(parents=True, exist_ok=True)

    api = PiperAPI(
        firmware=config.firmware,
        interface=config.interface,
        channel=config.channel,
        installation_pos=config.installation_pos,
    )
    pedal = FootPedalTrigger(
        config.foot_pedal_device,
        key_code=config.foot_pedal_key_code,
        debounce_s=config.foot_pedal_debounce_seconds,
    )
    camera = RealSenseStream(config)
    hand = None
    try:
        print("Connecting Piper...")
        api.connect()
        print("Piper connected:", api.is_ok(), "receive FPS:", api.get_fps())

        print("Connecting Midas hand...")
        hand = connect_hand(config, enable_torque=config.enable_hand_torque)
        print(f"Midas hand connected on {hand.port}; motor_ids={hand.motor_ids}")
        if not config.enable_hand_torque:
            print("Midas hand torque is disabled by this collector.")

        print("Opening foot pedal...")
        pedal.start()
        print(f"Foot pedal input: {pedal.active_device}")

        camera.start()
        print(f"Writing dataset under {dataset_root}")

        while True:
            wait_for_teacher_mode(api, poll_s=config.teacher_mode_poll_seconds)
            wait_for_foot_pedal(pedal, "Press the foot pedal to start recording.")
            if not wait_ready_delay(config, api):
                continue

            drained = camera.drain_pending_frames()
            if drained:
                print(f"Drained {drained} queued RealSense frames before recording.")

            episode_index = next_episode_index(raw_root(dataset_root))
            paths = episode_paths(dataset_root, episode_index)
            record_episode(
                paths=paths,
                config=config,
                camera=camera,
                hand=hand,
                api=api,
                pedal=pedal,
            )
            print("Episode complete. Waiting for the next teacher-mode episode.")
    except KeyboardInterrupt:
        print("\nCollection stopped.")
        return 0
    finally:
        try:
            camera.stop()
        finally:
            pedal.stop()
            api.disconnect()
            if hand is not None:
                hand.shutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
