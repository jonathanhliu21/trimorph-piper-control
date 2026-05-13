import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW


def main():
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=PiperFW.DEFAULT,  # Change if your firmware requires V183 or V188
        interface="socketcan",
        channel="can0",
    )

    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()

    try:
        time.sleep(0.5)

        print("Connected OK:", robot.is_ok())
        print("Receive FPS:", robot.get_fps())

        while True:
            joints = robot.get_joint_angles()    
            flange = robot.get_flange_pose()

            if joints is not None:
                print("joint angles [rad]:", joints.msg)
                print("joint timestamp:", joints.timestamp, "hz:", joints.hz)

            if flange is not None:
                x, y, z, roll, pitch, yaw = flange.msg
                print("flange pose [m, rad]:", flange.msg)
                print(f"x={x:.4f}, y={y:.4f}, z={z:.4f}, rpy=({roll:.4f}, {pitch:.4f}, {yaw:.4f})")
                print("flange timestamp:", flange.timestamp, "hz:", flange.hz)

            print()
            time.sleep(0.05)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()                        