from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("ur5_follower")
@dataclass
class UR5FollowerConfig(RobotConfig):
    # IP address of the UR5 controller (e.g. "192.168.1.100")
    ip_address: str = "192.10.0.11"

    # RTDE communication frequency in Hz
    rtde_frequency: float = 125.0

    # servoL parameters
    servo_speed: float = 0.5  # tool speed [m/s]
    servo_acceleration: float = 0.5  # tool acceleration [m/s^2]
    servo_lookahead_time: float = 0.1  # lookahead time [s] (0.03-0.2)
    servo_gain: float = 300  # servo gain (100-2000)

    # Robotiq Hand-E gripper settings
    use_gripper: bool = True
    gripper_speed: int = 255  # 0-255
    gripper_force: int = 50  # 0-255

    # Initial joint positions (in degrees) for optional homing.
    # Order: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
    # Defaults to None so teleoperation does not move the robot unless the user
    # explicitly configures a known-safe home pose for their cell.
    initial_joint_positions: list[float] | None = None

    # moveJ parameters for homing
    initial_move_speed: float = 0.5  # joint speed [rad/s]
    initial_move_acceleration: float = 0.5  # joint acceleration [rad/s^2]

    # Cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
