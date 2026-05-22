import argparse
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show a flipped Intel RealSense color camera view."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="RGB stream width. Default: 640",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="RGB stream height. Default: 480",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="RGB stream FPS. Default: 30",
    )
    parser.add_argument(
        "--flip",
        choices=("horizontal", "vertical", "both", "none"),
        default="both",
        help="Display flip direction. Default: both",
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
        help="List RealSense devices and exit.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List RGB color stream profiles for the selected RealSense camera and exit.",
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
            "for your Python environment."
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


def color_profiles(rs, device):
    profiles = []
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != rs.stream.color:
                continue
            video_profile = profile.as_video_stream_profile()
            if video_profile.format() != rs.format.rgb8:
                continue
            profiles.append(
                {
                    "width": video_profile.width(),
                    "height": video_profile.height(),
                    "fps": video_profile.fps(),
                }
            )
    profiles.sort(
        key=lambda item: (
            item["width"] * item["height"],
            item["fps"],
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
            f"rgb8 @{profile['fps']}fps"
        )


def profile_matches(profile, args):
    if args.width and profile["width"] != args.width:
        return False
    if args.height and profile["height"] != args.height:
        return False
    if args.fps and profile["fps"] != args.fps:
        return False
    return True


def choose_color_profile(rs, device, args):
    matching_profiles = [
        profile for profile in color_profiles(rs, device) if profile_matches(profile, args)
    ]
    if not matching_profiles:
        raise RuntimeError(
            "No matching color stream profile found. "
            "Run with --list-profiles to see supported modes."
        )

    target_pixels = args.width * args.height
    target_fps = args.fps

    return min(
        matching_profiles,
        key=lambda profile: (
            abs(profile["width"] * profile["height"] - target_pixels),
            abs(profile["fps"] - target_fps),
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


def main():
    args = parse_args()
    if args.width <= 0:
        raise ValueError("--width must be greater than 0")
    if args.height <= 0:
        raise ValueError("--height must be greater than 0")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0")
    if args.timeout_ms <= 0:
        raise ValueError("--timeout-ms must be greater than 0")
    if args.max_timeouts < 0:
        raise ValueError("--max-timeouts must be 0 or greater")

    cv2, np = import_camera_dependencies()
    rs = import_realsense_dependency()
    if args.list_devices:
        list_devices(rs)
        return

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
    selected = choose_color_profile(rs, device, args)
    print(
        "Selected color profile: "
        f"{selected['width']}x{selected['height']} rgb8 @{selected['fps']}fps"
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(
        rs.stream.color,
        selected["width"],
        selected["height"],
        rs.format.rgb8,
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

            rgb_image = np.asanyarray(color_frame.get_data())
            flipped_rgb_image = flip_frame(cv2, rgb_image, args.flip)
            cv2.imshow(
                args.window_name,
                cv2.cvtColor(flipped_rgb_image, cv2.COLOR_RGB2BGR),
            )

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
