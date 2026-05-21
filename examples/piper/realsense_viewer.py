import argparse
import glob
import sys
import time
from pathlib import Path


FORMAT_NAMES = ("auto", "bgr8", "rgb8", "yuyv", "mjpeg")
FORMAT_PREFERENCE = ("bgr8", "rgb8", "yuyv", "mjpeg")
V4L2_FORMATS = ("auto", "mjpg", "yuyv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show a flipped Intel RealSense color camera view."
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "v4l2", "realsense"),
        default="auto",
        help="Camera backend. auto tries the Cheese-style V4L2 path first. Default: auto",
    )
    parser.add_argument(
        "--camera-device",
        default=None,
        help="V4L2 camera device, for example /dev/video2. Default: auto-scan.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="OpenCV camera index to try before auto-scan, for example 0.",
    )
    parser.add_argument(
        "--v4l2-format",
        choices=V4L2_FORMATS,
        default="mjpg",
        help="Requested V4L2 pixel format. Default: mjpg",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Color stream width. Use 0 for auto. Default: 640",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Color stream height. Use 0 for auto. Default: 480",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Color stream FPS. Use 0 for auto. Default: 30",
    )
    parser.add_argument(
        "--format",
        choices=FORMAT_NAMES,
        default="auto",
        help="Color stream format. Default: auto",
    )
    parser.add_argument(
        "--flip",
        choices=("horizontal", "vertical", "both", "none"),
        default="both",
        help="Display flip direction. Default: horizontal",
    )
    parser.add_argument(
        "--serial",
        default=None,
        help="RealSense camera serial number to use when multiple cameras are connected.",
    )
    parser.add_argument(
        "--window-name",
        default="RealSense flipped view",
        help="OpenCV window title. Default: RealSense flipped view",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=1000,
        help="Milliseconds to wait for each frame before retrying. Default: 1000",
    )
    parser.add_argument(
        "--max-timeouts",
        type=int,
        default=0,
        help="Stop after this many frame timeouts. Use 0 to keep waiting. Default: 0",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List camera devices and exit.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List color stream profiles for the selected RealSense camera and exit.",
    )
    return parser.parse_args()


def import_camera_dependencies():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing NumPy. Install with: python3 -m pip install numpy"
        ) from exc

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing OpenCV Python bindings. Install with: python3 -m pip install opencv-python"
        ) from exc

    return cv2, np


def import_realsense_dependency():
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Intel RealSense Python bindings. Install librealsense/pyrealsense2 "
            "for your Python environment, or use --backend v4l2."
        ) from exc
    return rs


def flip_frame(cv2, frame, flip_mode):
    if flip_mode == "horizontal":
        return cv2.flip(frame, 1)
    if flip_mode == "vertical":
        return cv2.flip(frame, 0)
    if flip_mode == "both":
        return cv2.flip(frame, -1)
    return frame


def rs_format(rs, format_name):
    return getattr(rs.format, format_name)


def format_name(rs, stream_profile):
    video_profile = stream_profile.as_video_stream_profile()
    for name in FORMAT_NAMES:
        if name == "auto":
            continue
        if video_profile.format() == rs_format(rs, name):
            return name
    return str(video_profile.format())


def color_profiles(rs, device):
    profiles = []
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != rs.stream.color:
                continue
            video_profile = profile.as_video_stream_profile()
            profiles.append(
                {
                    "width": video_profile.width(),
                    "height": video_profile.height(),
                    "fps": video_profile.fps(),
                    "format": format_name(rs, profile),
                }
            )
    profiles.sort(
        key=lambda item: (
            item["width"] * item["height"],
            item["fps"],
            item["format"],
        )
    )
    return profiles


def print_color_profiles(rs, device):
    profiles = color_profiles(rs, device)
    if not profiles:
        print("No color stream profiles found.")
        return
    for profile in profiles:
        print(
            f"{profile['width']}x{profile['height']} "
            f"{profile['format']} @{profile['fps']}fps"
        )


def profile_matches(profile, args):
    if args.width and profile["width"] != args.width:
        return False
    if args.height and profile["height"] != args.height:
        return False
    if args.fps and profile["fps"] != args.fps:
        return False
    if args.format != "auto" and profile["format"] != args.format:
        return False
    return profile["format"] in FORMAT_PREFERENCE


def frame_to_bgr(cv2, np, color_frame, format_name):
    if format_name == "mjpeg":
        encoded = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode MJPEG color frame")
        return image

    image = np.asanyarray(color_frame.get_data())
    if format_name == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if format_name == "yuyv":
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)
    return image


def choose_color_profile(rs, device, args, usb_type):
    matching_profiles = [
        profile for profile in color_profiles(rs, device) if profile_matches(profile, args)
    ]
    if not matching_profiles:
        raise RuntimeError(
            "No matching color stream profile found. "
            "Run with --list-profiles to see supported modes."
        )

    preferred_formats = {name: index for index, name in enumerate(FORMAT_PREFERENCE)}
    target_pixels = None
    target_fps = None
    if args.width and args.height:
        target_pixels = args.width * args.height
    if args.fps:
        target_fps = args.fps

    if target_pixels is None:
        target_pixels = 640 * 360 if usb_type.startswith("2") else 640 * 480
    if target_fps is None:
        target_fps = 15 if usb_type.startswith("2") else 30

    return min(
        matching_profiles,
        key=lambda profile: (
            abs(profile["width"] * profile["height"] - target_pixels),
            abs(profile["fps"] - target_fps),
            preferred_formats.get(profile["format"], len(FORMAT_PREFERENCE)),
        ),
    )


def device_info(rs, device):
    def get_info(info):
        if not device.supports(info):
            return ""
        return device.get_info(info)

    name = get_info(rs.camera_info.name) or "Unknown RealSense"
    serial = get_info(rs.camera_info.serial_number) or "unknown-serial"
    firmware = get_info(rs.camera_info.firmware_version) or "unknown-firmware"
    usb_type = get_info(rs.camera_info.usb_type_descriptor) or "unknown-usb"
    return name, serial, firmware, usb_type


def list_devices(rs):
    context = rs.context()
    devices = context.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
        return

    for index, device in enumerate(devices):
        name, serial, firmware, usb_type = device_info(rs, device)
        print(f"[{index}] {name}")
        print(f"  serial: {serial}")
        print(f"  firmware: {firmware}")
        print(f"  usb: {usb_type}")
        for sensor in device.query_sensors():
            print(f"  sensor: {sensor.get_info(rs.camera_info.name)}")


def video_device_name(device_path):
    name_path = Path("/sys/class/video4linux") / Path(device_path).name / "name"
    try:
        return name_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def list_video_devices():
    devices = sorted(glob.glob("/dev/video*"))
    if not devices:
        print("No /dev/video* devices found.")
        return

    for device_path in devices:
        name = video_device_name(device_path) or "(unknown name)"
        print(f"{device_path}: {name}")


def v4l2_sources(args):
    sources = []
    if args.camera_device is not None:
        sources.append(args.camera_device)
    if args.camera_index is not None:
        sources.append(args.camera_index)
    sources.extend(sorted(glob.glob("/dev/video*"), key=video_device_sort_key))

    deduped = []
    seen = set()
    for source in sources:
        key = str(source)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def video_device_sort_key(path):
    name = Path(path).name
    try:
        return int(name.replace("video", ""))
    except ValueError:
        return name


def set_capture_format(cv2, capture, args):
    if args.v4l2_format == "mjpg":
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    elif args.v4l2_format == "yuyv":
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

    if args.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        capture.set(cv2.CAP_PROP_FPS, args.fps)


def describe_capture(cv2, capture):
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fourcc_text = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip()
    return f"{width}x{height} {fourcc_text or 'unknown-format'} @{fps:.1f}fps"


def open_v4l2_capture(cv2, args):
    backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
    errors = []
    for source in v4l2_sources(args):
        capture = cv2.VideoCapture(source, backend)
        if not capture.isOpened():
            capture.release()
            errors.append(f"{source}: failed to open")
            continue

        set_capture_format(cv2, capture, args)
        ok, frame = capture.read()
        if ok and frame is not None:
            return source, capture, frame

        capture.release()
        errors.append(f"{source}: opened but no frame")

    detail = "; ".join(errors) if errors else "no candidate /dev/video* devices"
    raise RuntimeError(f"Could not open a V4L2 camera frame ({detail})")


def find_device(rs, serial=None):
    context = rs.context()
    devices = context.query_devices()
    for device in devices:
        _, device_serial, _, _ = device_info(rs, device)
        if serial is None or serial == device_serial:
            return device
    if serial:
        raise RuntimeError(f"No RealSense device found with serial {serial}")
    raise RuntimeError("No RealSense devices found")


def run_v4l2_viewer(cv2, args):
    print("Starting V4L2 camera stream. Press q or Esc to quit.")
    source, capture, first_frame = open_v4l2_capture(cv2, args)
    name = video_device_name(source) if isinstance(source, str) else ""
    label = f"{source}"
    if name:
        label = f"{label} ({name})"
    print(f"Using {label}")
    print(f"Active capture: {describe_capture(cv2, capture)}")

    frame = first_frame
    try:
        while True:
            image = flip_frame(cv2, frame, args.flip)
            cv2.imshow(args.window_name, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            ok, frame = capture.read()
            if not ok or frame is None:
                print("Camera read failed; waiting for the next frame...")
                time.sleep(0.05)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("V4L2 camera stopped delivering frames")
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    if args.width < 0:
        raise ValueError("--width must be 0 or greater")
    if args.height < 0:
        raise ValueError("--height must be 0 or greater")
    if bool(args.width) != bool(args.height):
        raise ValueError("--width and --height must both be set, or both be 0 for auto")
    if args.fps < 0:
        raise ValueError("--fps must be 0 or greater")
    if args.timeout_ms <= 0:
        raise ValueError("--timeout-ms must be greater than 0")
    if args.max_timeouts < 0:
        raise ValueError("--max-timeouts must be 0 or greater")

    cv2, np = import_camera_dependencies()
    if args.list_devices:
        print("V4L2 devices:")
        list_video_devices()
        if args.backend in ("auto", "realsense"):
            print()
            print("RealSense SDK devices:")
            try:
                list_devices(import_realsense_dependency())
            except RuntimeError as exc:
                print(exc)
        return

    if args.backend in ("auto", "v4l2") and not args.list_profiles:
        try:
            run_v4l2_viewer(cv2, args)
            return
        except RuntimeError as exc:
            if args.backend == "v4l2":
                raise
            print(f"V4L2 backend failed: {exc}")
            print("Falling back to RealSense SDK backend.")

    rs = import_realsense_dependency()
    run_realsense_viewer(cv2, np, rs, args)


def run_realsense_viewer(cv2, np, rs, args):
    device = find_device(rs, args.serial)
    name, serial, firmware, usb_type = device_info(rs, device)
    if args.list_profiles:
        print(f"{name} serial={serial} firmware={firmware} usb={usb_type}")
        print_color_profiles(rs, device)
        return

    if usb_type.startswith("2"):
        print(
            "Warning: camera reports USB 2.x. "
            "RealSense D435 is much more reliable on a USB 3 port/cable."
        )
    selected = choose_color_profile(rs, device, args, usb_type)
    print(
        "Selected color profile: "
        f"{selected['width']}x{selected['height']} {selected['format']} @{selected['fps']}fps"
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(
        rs.stream.color,
        selected["width"],
        selected["height"],
        rs_format(rs, selected["format"]),
        selected["fps"],
    )

    print("Starting RealSense color stream. Press q or Esc to quit.")
    profile = pipeline.start(config)
    active_name, active_serial, active_firmware, active_usb_type = device_info(
        rs,
        profile.get_device(),
    )
    print(
        f"Using {active_name} serial={active_serial} "
        f"firmware={active_firmware} usb={active_usb_type}"
    )

    timeout_count = 0
    next_timeout_notice_t = 0.0
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(args.timeout_ms)
            except RuntimeError as exc:
                if "Frame didn't arrive" not in str(exc):
                    raise
                timeout_count += 1
                now = time.monotonic()
                if now >= next_timeout_notice_t:
                    print(
                        "Still waiting for color frames... "
                        "Check that the camera is not open in another app and try a lower FPS/resolution."
                    )
                    next_timeout_notice_t = now + 2.0
                if args.max_timeouts and timeout_count >= args.max_timeouts:
                    raise RuntimeError(
                        f"No frames arrived after {timeout_count} timeouts"
                    ) from exc
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            timeout_count = 0

            image = frame_to_bgr(cv2, np, color_frame, selected["format"])
            image = flip_frame(cv2, image, args.flip)
            cv2.imshow(args.window_name, image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
