"""Print the phone's live 6D Teledex pose.

This is a read-only diagnostic sibling to collect_with_home.py: it connects to
the same phone pose stream, waits for AR tracking to become available, then
prints x/y/z plus XYZ Euler angles until Ctrl+C.
"""
import argparse
import time

from scipy.spatial.transform import Rotation as R

from phone import Phone


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Continuously print the phone's Teledex 6D pose."
    )
    parser.add_argument("--frequency", type=float, default=10.0,
                        help="Print frequency in Hz. Default: 10")
    parser.add_argument("--hand", action="store_true",
                        help="Print the hand-tracked pose instead of the phone pose")
    parser.add_argument("--degrees", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Print roll/pitch/yaw in degrees. Default: true")
    return parser.parse_args()


def _format_pose(position, rotation, *, degrees=True):
    euler = R.from_matrix(rotation).as_euler("xyz", degrees=degrees)
    units = "deg" if degrees else "rad"
    return (
        f"x={position[0]:+.6f} y={position[1]:+.6f} z={position[2]:+.6f} "
        f"roll={euler[0]:+.3f} pitch={euler[1]:+.3f} yaw={euler[2]:+.3f} ({units})"
    )


def _button_states(phone):
    return {
        "primary": bool(phone.get_button_state()),
        "secondary": bool(phone.get_button_secondary_state()),
        "toggle": bool(phone.get_grasp_state()),
    }


def main():
    args = _parse_args()
    interval = 1.0 / args.frequency if args.frequency > 0 else 0.0

    phone = Phone()
    if args.hand:
        phone.hand = 1
        phone.position_name = "position_hand"
        phone.rotation_name = "rotation_hand"

    print("Press Ctrl+C to quit.")
    previous_buttons = _button_states(phone)

    try:
        while True:
            buttons = _button_states(phone)
            clicked = [
                name for name, pressed in buttons.items()
                if pressed and not previous_buttons.get(name, False)
            ]
            if clicked:
                print(f"button clicked: {', '.join(clicked)}", flush=True)
                if "primary" in clicked:
                    break
            previous_buttons = buttons

            position, rotation = phone.get_position_rotation()
            print(_format_pose(position, rotation, degrees=args.degrees),
                  flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nQuitting...")
    finally:
        stop = getattr(phone.session, "stop", None)
        if callable(stop):
            stop()


if __name__ == "__main__":
    main()
