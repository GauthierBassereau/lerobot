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
    # First recording session (creates dataset from scratch):
    python record.py

    # Resume recording (appends to existing dataset):
    python record.py --resume

    # Record evaluation episodes into a separate dataset:
    python record.py --split eval

    # Push to HuggingFace Hub after recording:
    python record.py --resume --push
"""

import argparse

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.feature_utils import combine_feature_dicts
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
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

NUM_EPISODES = 50
FPS = 30
EPISODE_TIME_SEC = 600  # Max time per episode — press Right Arrow to end episode early whenever you want
RESET_TIME_SEC = 20
TASK = "Interact with objects on the table"
BASE_HF_REPO_ID = "Gaugou/ur5"
# Keep visualization off by default while recording. It is useful for debugging,
# but it adds CPU work to the control loop and can increase jitter.
DISPLAY_DATA = False
# Start with threads only for a single camera. Extra processes add pickling/copy
# overhead for image payloads and usually hurt latency-sensitive recording loops.
IMAGE_WRITER_PROCESSES = 0
IMAGE_WRITER_THREADS = 2
# Additional recording/encoding knobs that affect performance independently from
# the PNG image writer settings above.
VIDEO_ENCODING_BATCH_SIZE = 1
VIDEO_CODEC = "auto"
STREAMING_ENCODING = False
ENCODER_QUEUE_MAXSIZE = 30
ENCODER_THREADS = 1
EE_STEP_SIZES = {"x": 0.7, "y": 0.7, "z": 0.7}
EE_BOUNDS = {
    "min": [-0.8, -0.8, 0.05],
    "max": [0.8, 0.8, 0.8],
}
MAX_EE_STEP_M = 0.06


def get_repo_id(base_repo_id: str, split: str) -> str:
    if split == "train":
        return base_repo_id
    return f"{base_repo_id}_{split}"


def main():
    # ======================== CLI Arguments ========================
    parser = argparse.ArgumentParser(description="Phone → UR5 Dataset Recording")
    parser.add_argument("--resume", action="store_true", help="Resume recording on an existing dataset")
    parser.add_argument("--push", action="store_true", help="Push dataset to HuggingFace Hub after recording")
    parser.add_argument(
        "--split",
        choices=("train", "eval"),
        default="train",
        help="Dataset split to record. 'eval' is stored in a separate dataset repo.",
    )
    args = parser.parse_args()
    repo_id = get_repo_id(BASE_HF_REPO_ID, args.split)

    # ======================== Configuration ========================
    # Update the IP address to match your UR5 controller
    # Update the RealSense serial number to match your camera
    camera_config = {"front": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS)}
    robot_config = UR5FollowerConfig(
        ip_address="192.10.0.11",
        id="my_ur5",
        cameras=camera_config,
        use_gripper=True,
        initial_joint_positions=None,
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
                end_effector_step_sizes=EE_STEP_SIZES,
                use_latched_reference=True,
            ),
            # EEBoundsAndSafety(
            #     end_effector_bounds=EE_BOUNDS,
            #     max_ee_step_m=MAX_EE_STEP_M,
            # ),
            GripperVelocityToJoint(speed_factor=100.0),
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
    dataset_features = combine_feature_dicts(
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
    )

    if args.resume:
        # Resume: load existing dataset and continue recording
        dataset = LeRobotDataset(
            repo_id=repo_id,
            batch_encoding_size=VIDEO_ENCODING_BATCH_SIZE,
            vcodec=VIDEO_CODEC,
            streaming_encoding=STREAMING_ENCODING,
            encoder_queue_maxsize=ENCODER_QUEUE_MAXSIZE,
            encoder_threads=ENCODER_THREADS,
        )
        dataset.start_image_writer(
            num_processes=IMAGE_WRITER_PROCESSES,
            num_threads=IMAGE_WRITER_THREADS,
        )
        print(f"\n✅ Resumed dataset '{repo_id}' ({args.split} split) with {dataset.num_episodes} existing episodes.")
    else:
        # Create new dataset from scratch
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=FPS,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_processes=IMAGE_WRITER_PROCESSES,
            image_writer_threads=IMAGE_WRITER_THREADS,
            batch_encoding_size=VIDEO_ENCODING_BATCH_SIZE,
            vcodec=VIDEO_CODEC,
            streaming_encoding=STREAMING_ENCODING,
            encoder_queue_maxsize=ENCODER_QUEUE_MAXSIZE,
            encoder_threads=ENCODER_THREADS,
        )
        print(f"\n✅ Created new dataset '{repo_id}' for the {args.split} split.")

    # ======================== Connect ========================
    robot_connected = False
    phone_connected = False
    listener = None
    try:
        phone.connect()
        phone_connected = True
        robot.connect()
        robot_connected = True

        listener, events = init_keyboard_listener()
        if DISPLAY_DATA:
            init_rerun(session_name="phone_ur5_record")

        _ = robot.get_observation()
        if not phone.is_connected:
            raise ValueError("Teleop is not connected!")

        # ======================== Recording Loop ========================
        start_episode = dataset.num_episodes  # episodes already in the dataset

        print(f"\n{'='*60}")
        print(f"  RECORDING SESSION")
        print(f"  Dataset: {repo_id}")
        print(f"  Split: {args.split}")
        print(f"  Existing episodes: {start_episode}")
        print(f"  Episodes to record: {NUM_EPISODES}")
        print(f"  FPS: {FPS} | Max episode time: {EPISODE_TIME_SEC}s")
        print(f"{'='*60}")
        print(f"\n  Controls:")
        print(f"    Hold B1 in the HEBI app to enable control")
        print(f"    Press B2 to toggle the gripper open/close")
        print(f"    → (Right Arrow) = End episode early")
        print(f"    ← (Left Arrow)  = Discard & re-record episode")
        print(f"    Esc              = Stop recording entirely")
        print(f"{'='*60}\n")

        session_episode = 0
        while session_episode < NUM_EPISODES and not events["stop_recording"]:
            total_episode = start_episode + session_episode
            log_say(f"Recording episode {total_episode + 1} (session {session_episode + 1} of {NUM_EPISODES})")

            # Reset the teleop action processor to clear old latched reference poses
            phone_to_robot_ee_pose_processor.reset()

            # Move to initial pose before each episode
            if robot.config.initial_joint_positions is not None:
                log_say("Moving to initial pose")
                robot.move_to_initial_pose()

            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=phone,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK,
                display_data=DISPLAY_DATA,
                teleop_action_processor=phone_to_robot_ee_pose_processor,
                robot_action_processor=robot_ee_to_ur5_processor,
                robot_observation_processor=robot_obs_to_ee_processor,
            )

            # Reset environment
            if not events["stop_recording"] and (session_episode < NUM_EPISODES - 1 or events["rerecord_episode"]):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop=phone,
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK,
                    display_data=DISPLAY_DATA,
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
            session_episode += 1
            log_say(f"Episode saved. Total episodes in dataset: {dataset.num_episodes}")

    finally:
        # ======================== Cleanup (always runs) ========================
        log_say("Stop recording")
        if robot_connected:
            robot.disconnect()
        if phone_connected:
            phone.disconnect()
        if listener is not None:
            listener.stop()

        dataset.finalize()
        print(f"\n✅ Dataset finalized with {dataset.num_episodes} total episodes.")

        if args.push:
            print("Pushing to HuggingFace Hub...")
            dataset.push_to_hub()
            print("✅ Pushed to Hub.")
        else:
            split_args = "" if args.split == "train" else f" --split {args.split}"
            print(f"💡 To push to Hub later, run: python record.py --resume{split_args} --push")


if __name__ == "__main__":
    main()
