"""
Phone -> UR5 teleoperation example.

Teleoperates a UR5 robot arm (with optional Robotiq Hand-E gripper) using an
iPhone via the HEBI Mobile I/O app. The UR5 receives Cartesian end-effector
commands via servoL, so no URDF or external IK solver is needed.

Usage:
    python teleoperate.py
"""

import time

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    GripperVelocityToJoint,
)
from lerobot.robots.ur5_follower.config_ur5_follower import UR5FollowerConfig
from lerobot.robots.ur5_follower.ur5_follower import UR5Follower
from lerobot.robots.ur5_follower.ur5_processor import EEReferenceAndDeltaFromTCP, EEToUR5Action
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30
EE_STEP_SIZES = {"x": 0.6, "y": 0.6, "z": 0.6}
EE_BOUNDS = {
    "min": [-0.8, -0.8, -0.2],
    "max": [0.8, 0.8, 0.8],
}
MAX_EE_STEP_M = 0.06


def main():
    camera_config = {"front": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS)}
    robot_config = UR5FollowerConfig(
        id="my_ur5",
        cameras=camera_config,
        use_gripper=True,
        initial_joint_positions=None,
    )
    teleop_config = PhoneConfig(phone_os=PhoneOS.IOS)

    robot = UR5Follower(robot_config)
    teleop_device = Phone(teleop_config)

    phone_to_ur5_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
            EEReferenceAndDeltaFromTCP(
                end_effector_step_sizes=EE_STEP_SIZES,
                use_latched_reference=True,
            ),
            # EEBoundsAndSafety(
            #     end_effector_bounds=EE_BOUNDS,
            #     max_ee_step_m=MAX_EE_STEP_M,
            # ),
            GripperVelocityToJoint(speed_factor=100.0),
            EEToUR5Action(),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    robot_connected = False
    teleop_connected = False
    try:
        teleop_device.connect()
        teleop_connected = True
        robot.connect()
        robot_connected = True

        init_rerun(session_name="phone_ur5_teleop")

        _ = robot.get_observation()
        if not teleop_device.is_connected:
            raise ValueError("Teleop is not connected!")

        print("Starting teleop loop. Move your phone to teleoperate the UR5...")
        print("Hold B1 in the HEBI app to enable control.")
        print("Press B2 to toggle the gripper open/close.")

        while True:
            t0 = time.perf_counter()

            robot_obs = robot.get_observation()

            phone_action = teleop_device.get_action()
            if not phone_action:
                precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
                continue

            ur5_action = phone_to_ur5_processor((phone_action, robot_obs))
            _ = robot.send_action(ur5_action)

            log_rerun_data(observation={**robot_obs, **phone_action}, action=ur5_action)
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt! Stopping gracefully...")
    finally:
        if robot_connected:
            robot.disconnect()
        if teleop_connected:
            teleop_device.disconnect()
        print("Disconnected successfully.")


if __name__ == "__main__":
    main()
