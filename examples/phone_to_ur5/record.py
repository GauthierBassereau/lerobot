"""
Phone → UR5 Dataset Recording Example

Records teleoperation episodes of a UR5 robot arm (with Robotiq Hand-E gripper)
controlled by an iPhone via the HEBI Mobile I/O app.

The dataset captures:
- Actions: EE pose targets (x, y, z, wx, wy, wz) + gripper position
- Observations: EE pose (from UR5 TCP) + gripper + camera images

Pipeline:
    Phone 6-DoF → MapPhoneAction → EEReferenceAndDeltaFromTCP → EEBoundsAndSafety
    → GripperVelToJoint → EEToUR5Action → servoL (RTDE)

Prerequisites:
    pip install "lerobot[phone,ur5]"

Usage:
    python record.py
"""

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    GripperVelocityToJoint,
)
from lerobot.robots.ur5_follower.config_ur5_follower import UR5FollowerConfig
from lerobot.robots.ur5_follower.ur5_follower import UR5Follower
from lerobot.robots.ur5_follower.ur5_processor import (
    EEReferenceAndDeltaFromTCP,
    EEToUR5Action,
    UR5ObservationToEE,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 30
TASK_DESCRIPTION = "Pick and place the object"
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"

# ======================== Configuration ========================
# Update the IP address to match your UR5 controller
# Update the RealSense serial number to match your camera
camera_config = {"front": RealSenseCameraConfig(serial_number_or_name="YOUR_SERIAL_NUMBER", width=640, height=480, fps=FPS)}
robot_config = UR5FollowerConfig(
    ip_address="192.168.1.100",
    id="my_ur5",
    cameras=camera_config,
    use_gripper=True,
)
teleop_config = PhoneConfig(phone_os=PhoneOS.IOS)

# ======================== Initialization ========================
robot = UR5Follower(robot_config)
phone = Phone(teleop_config)

# ======================== Processor Pipelines ========================

# Pipeline 1: Phone → EE pose action (for teleoperation + dataset action recording)
# EEReferenceAndDeltaFromTCP reads UR5 TCP pose directly — no URDF/FK needed
phone_to_robot_ee_pose_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
    steps=[
        MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
        EEReferenceAndDeltaFromTCP(
            end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},
            use_latched_reference=True,
        ),
        EEBoundsAndSafety(
            end_effector_bounds={
                "min": [-1.0, -1.0, 0.0],
                "max": [1.0, 1.0, 1.5],
            },
            max_ee_step_m=0.10,
        ),
        GripperVelocityToJoint(speed_factor=20.0),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# Pipeline 2: EE action → UR5 action (pass-through for servoL)
robot_ee_to_ur5_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
    steps=[
        EEToUR5Action(),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# Pipeline 3: UR5 observation → EE observation (for dataset recording)
robot_obs_to_ee_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
    steps=[
        UR5ObservationToEE(keep_joints=True),
    ],
    to_transition=observation_to_transition,
    to_output=transition_to_observation,
)

# ======================== Dataset ========================
dataset = LeRobotDataset.create(
    repo_id=HF_REPO_ID,
    fps=FPS,
    features=combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=phone_to_robot_ee_pose_processor,
            initial_features=create_initial_features(action=phone.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_obs_to_ee_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    ),
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# ======================== Connect ========================
phone.connect()
robot.connect()

listener, events = init_keyboard_listener()
init_rerun(session_name="phone_ur5_record")

if not robot.is_connected or not phone.is_connected:
    raise ValueError("Robot or teleop is not connected!")

# ======================== Recording Loop ========================
print("Starting record loop. Move your phone to teleoperate the UR5...")
print("Hold B1 in the HEBI app to enable control.")
print("Press B2 to toggle the gripper open/close.")
print("Press 'q' to stop recording, 'r' to re-record episode.")

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop=phone,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        teleop_action_processor=phone_to_robot_ee_pose_processor,
        robot_action_processor=robot_ee_to_ur5_processor,
        robot_observation_processor=robot_obs_to_ee_processor,
    )

    # Reset environment
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=phone,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=phone_to_robot_ee_pose_processor,
            robot_action_processor=robot_ee_to_ur5_processor,
            robot_observation_processor=robot_obs_to_ee_processor,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# ======================== Cleanup ========================
log_say("Stop recording")
robot.disconnect()
phone.disconnect()
listener.stop()

dataset.finalize()
dataset.push_to_hub()
