"""
Phone → UR5 Teleoperation Example

Teleoperates a UR5 robot arm (with Robotiq Hand-E gripper) using an iPhone
via the HEBI Mobile I/O app. The UR5 receives Cartesian end-effector commands
via servoL — no URDF or external IK solver needed.

Pipeline:
    Phone 6-DoF → MapPhoneAction → EEReferenceAndDeltaFromTCP → EEBoundsAndSafety
    → GripperVelToJoint → EEToUR5Action → servoL (RTDE)

Prerequisites:
    pip install "lerobot[phone,ur5]"

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
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    GripperVelocityToJoint,
)
from lerobot.robots.ur5_follower.config_ur5_follower import UR5FollowerConfig
from lerobot.robots.ur5_follower.ur5_follower import UR5Follower
from lerobot.robots.ur5_follower.ur5_processor import EEReferenceAndDeltaFromTCP, EEToUR5Action
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

# ======================== Configuration ========================
# Update the IP address to match your UR5 controller
# Update the RealSense serial number to match your camera
camera_config = {"front": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS)}
robot_config = UR5FollowerConfig(
    id="my_ur5",
    cameras=camera_config,
    use_gripper=True,
)
teleop_config = PhoneConfig(phone_os=PhoneOS.IOS)

# ======================== Initialization ========================
robot = UR5Follower(robot_config)
teleop_device = Phone(teleop_config)

# Build pipeline: Phone → EE pose → UR5 servoL
# NOTE: Unlike SO100, no InverseKinematics or URDF needed.
# EEReferenceAndDeltaFromTCP reads the UR5's TCP pose directly from observations.
phone_to_ur5_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
    steps=[
        MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
        EEReferenceAndDeltaFromTCP(
            end_effector_step_sizes={"x": 0.7, "y": 0.7, "z": 0.7},
            use_latched_reference=True,
        ),
        # NOT NEEDED UR5 has its own safety bounds
        # EEBoundsAndSafety(
        #     # UR5 workspace bounds in meters — adjust for your setup
        #     end_effector_bounds={
        #         "min": [-1.0, -1.0, 0.0],  # Don't go below table
        #         "max": [1.0, 1.0, 1.5],
        #     },
        #     max_ee_step_m=0.1,  # Max 10cm per step
        # ),
        GripperVelocityToJoint(speed_factor=100.0),
        EEToUR5Action(),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# ======================== Connect ========================
teleop_device.connect()
robot.connect()
robot.move_to_initial_pose()

# Init rerun viewer
init_rerun(session_name="phone_ur5_teleop")

if not robot.is_connected or not teleop_device.is_connected:
    raise ValueError("Robot or teleop is not connected!")

print("Starting teleop loop. Move your phone to teleoperate the UR5...")
print("Hold B1 in the HEBI app to enable control.")
print("Press B2 to toggle the gripper open/close.")

# ======================== Main Loop ========================
try:
    while True:
        t0 = time.perf_counter()

        # Get robot observation (includes joints + TCP pose + gripper)
        robot_obs = robot.get_observation()

        # Get teleop action from phone
        phone_action = teleop_device.get_action()
        if not phone_action:
            busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
            continue

        # Phone → EE pose → UR5 action
        ur5_action = phone_to_ur5_processor((phone_action, robot_obs))

        # Send action to UR5 (servoL + gripper)
        _ = robot.send_action(ur5_action)

        # Visualize
        log_rerun_data(observation={**robot_obs, **phone_action}, action=ur5_action)

        busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
except KeyboardInterrupt:
    print("\nCaught KeyboardInterrupt! Stopping gracefully...")
finally:
    robot.disconnect()
    teleop_device.disconnect()
    print("Disconnected successfully.")
