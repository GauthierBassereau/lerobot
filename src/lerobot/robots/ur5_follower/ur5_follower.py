import logging
import time
import socket
import threading
from functools import cached_property
from typing import Any
from collections import OrderedDict

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_ur5_follower import UR5FollowerConfig

logger = logging.getLogger(__name__)

# UR5 joint names in order
UR5_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


class RobotiqGripperSocket:
    """Robotiq gripper control via URCap socket service (port 63352)."""

    ACT = "ACT"
    GTO = "GTO"
    ATR = "ATR"
    ADR = "ADR"
    FOR = "FOR"
    SPE = "SPE"
    POS = "POS"
    STA = "STA"
    PRE = "PRE"
    OBJ = "OBJ"
    FLT = "FLT"
    ENCODING = "UTF-8"

    def __init__(self, host: str, port: int = 63352, *, socket_timeout: float = 8.0):
        self.host = host
        self.port = int(port)
        self.socket_timeout = float(socket_timeout)
        self.socket: socket.socket | None = None
        self._lock = threading.Lock()
        self._rx_buf = bytearray()

    def connect(self) -> None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.socket_timeout)
            s.connect((self.host, self.port))
            self.socket = s
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Robotiq socket at {self.host}:{self.port}: {e}") from e

    def disconnect(self) -> None:
        if self.socket is None:
            return
        try:
            self.socket.close()
        finally:
            self.socket = None
            self._rx_buf.clear()

    def _recv_line(self) -> str:
        """Receive a single \\n-terminated line (stripped).
        
        Also handles responses without newline (e.g., "ack" without \\n).
        Uses short timeouts per recv() call to avoid blocking for full socket timeout.
        """
        if self.socket is None:
            raise ConnectionError("Robotiq socket not connected")
        
        # Check if we already have a complete line
        nl = self._rx_buf.find(b"\n")
        if nl != -1:
            line = bytes(self._rx_buf[:nl])
            del self._rx_buf[: nl + 1]
            return line.decode(self.ENCODING, errors="replace").strip()
        
        # Check if buffer contains "ack" without newline (common case)
        if len(self._rx_buf) > 0:
            buf_str = self._rx_buf.decode(self.ENCODING, errors="replace").strip().lower()
            if buf_str == "ack":
                self._rx_buf.clear()
                return "ack"
        
        # Use shorter timeout per recv() call to avoid blocking for full 8s
        recv_timeout = 0.5  # 500ms per recv() call
        original_timeout = self.socket.gettimeout()
        
        try:
            self.socket.settimeout(recv_timeout)
            t0 = time.time()
            grace_period = 0.1  # 100ms grace period for newline after receiving data
            
            while True:
                # Check for newline
                nl = self._rx_buf.find(b"\n")
                if nl != -1:
                    line = bytes(self._rx_buf[:nl])
                    del self._rx_buf[: nl + 1]
                    return line.decode(self.ENCODING, errors="replace").strip()
                
                # Check if buffer contains "ack" (with or without newline)
                if len(self._rx_buf) > 0:
                    buf_str = self._rx_buf.decode(self.ENCODING, errors="replace").strip().lower()
                    if "ack" in buf_str:
                        # If we have "ack" and grace period expired, return it
                        if time.time() - t0 > grace_period:
                            self._rx_buf.clear()
                            return "ack"
                
                # Try to receive more data with short timeout
                try:
                    chunk = self.socket.recv(1024)
                    if not chunk:
                        raise ConnectionError("Robotiq socket closed by peer")
                    self._rx_buf.extend(chunk)
                    t0 = time.time()  # Reset grace period when we receive data
                except socket.timeout:
                    # If we have "ack" in buffer and timeout, return it
                    if len(self._rx_buf) > 0:
                        buf_str = self._rx_buf.decode(self.ENCODING, errors="replace").strip().lower()
                        if "ack" in buf_str:
                            self._rx_buf.clear()
                            return "ack"
                    # Otherwise, raise timeout - caller can handle it
                    raise
        finally:
            # Restore original timeout
            try:
                self.socket.settimeout(original_timeout)
            except Exception:
                pass

    def _send_and_recv_line(self, cmd: str) -> str:
        if self.socket is None:
            raise ConnectionError("Robotiq socket not connected")
        with self._lock:
            self.socket.sendall(cmd.encode(self.ENCODING))
            return self._recv_line()

    @staticmethod
    def _is_ack(line: str) -> bool:
        return "ack" in line.strip().lower()

    def _set_vars(self, var_dict: "OrderedDict[str, int]") -> None:
        # We do not toggle GTO to 0 here because it pauses the gripper 
        # for >20ms every time a command is sent, causing severe latency.
        
        cmd = "SET"
        for variable, value in var_dict.items():
            cmd += f" {variable} {int(value)}"
        cmd += "\n"
        line = self._send_and_recv_line(cmd)
        if not self._is_ack(line):
            raise RuntimeError(f"Robotiq SET not acknowledged: {line!r}")

    def _set_var(self, variable: str, value: int) -> None:
        self._set_vars(OrderedDict([(variable, int(value))]))

    def _get_var(self, variable: str) -> int:
        line = self._send_and_recv_line(f"GET {variable}\n")
        parts = line.split()
        if len(parts) != 2:
            raise RuntimeError(f"Unexpected GET response: {line!r}")
        var_name, value_str = parts
        if var_name != variable:
            raise RuntimeError(f"Unexpected GET response {line!r}: expected '{variable}'")
        return int(value_str)

    def _reset(self) -> None:
        self._set_var(self.ACT, 0)
        self._set_var(self.ATR, 0)
        t0 = time.time()
        while time.time() - t0 < 5.0:
            if self._get_var(self.ACT) == 0 and self._get_var(self.STA) == 0:
                break
            self._set_var(self.ACT, 0)
            self._set_var(self.ATR, 0)
            time.sleep(0.1)
        time.sleep(0.5)

    def is_active(self) -> bool:
        return self._get_var(self.STA) == 3

    def activate(self) -> None:
        if self.is_active():
            return
        self._reset()
        self._set_var(self.ACT, 1)
        t0 = time.time()
        while time.time() - t0 < 10.0:
            if self._get_var(self.ACT) == 1 and self._get_var(self.STA) == 3:
                return
            time.sleep(0.1)
        raise RuntimeError("Robotiq activation timed out (STA did not reach 3)")

    def move_and_wait(self, position: int, speed: int = 128, force: int = 64) -> None:
        position = int(np.clip(position, 0, 255))
        speed = int(np.clip(speed, 0, 255))
        force = int(np.clip(force, 0, 255))
        
        self._set_vars(OrderedDict([(self.POS, position), (self.SPE, speed), (self.FOR, force), (self.GTO, 1)]))
        # For teleop we don't block and wait. We just send the command.
        # This was move_and_wait in the original script, but changed to not block for high frequency teleop.

    def get_position(self) -> int:
        return self._get_var(self.PRE)


class AsyncGripperWrapper:
    """Non-blocking wrapper around RobotiqGripperSocket.

    Runs all socket I/O in a daemon thread so the main teleop loop
    is never blocked by gripper communication.
    """

    def __init__(self, gripper: RobotiqGripperSocket):
        self._gripper = gripper
        self._lock = threading.Lock()
        self._cached_pos: float = 0.0
        self._target_cmd: tuple[int, int, int] | None = None  # (pos, speed, force)
        self._last_cmd: tuple[int, int, int] | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while self._running:
            try:
                # Send any pending command
                with self._lock:
                    cmd = self._target_cmd
                    self._target_cmd = None
                
                # Only send if the command has changed to avoid spamming the gripper socket
                if cmd is not None and cmd != self._last_cmd:
                    self._gripper.move_and_wait(*cmd)
                    self._last_cmd = cmd

                # Read position and cache it
                pos = self._gripper.get_position()
                with self._lock:
                    self._cached_pos = float(pos) / 255.0 * 100.0
            except Exception:
                pass  # Don't crash the thread on transient socket errors
            time.sleep(0.01)  # ~100 Hz attempt rate (actual rate limited by socket)

    def get_cached_position(self) -> float:
        """Returns last known gripper position (0-100), never blocks."""
        with self._lock:
            return self._cached_pos

    def send_command(self, position: int, speed: int, force: int) -> None:
        """Queue a gripper command, never blocks."""
        with self._lock:
            self._target_cmd = (position, speed, force)


class UR5Follower(Robot):
    """
    UR5 robot arm controlled via RTDE (Real-Time Data Exchange) protocol.
    Optionally equipped with a Robotiq Hand-E gripper.

    Uses servoL for Cartesian end-effector control, allowing the UR5's
    built-in IK solver to handle joint-space conversion.
    """

    config_class = UR5FollowerConfig
    name = "ur5_follower"

    def __init__(self, config: UR5FollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)

        self._rtde_control = None
        self._rtde_receive = None
        self._gripper = None
        self._async_gripper: AsyncGripperWrapper | None = None

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {}
        # Joint positions
        for name in UR5_JOINT_NAMES:
            features[f"{name}.pos"] = float
        # TCP pose (end-effector)
        for k in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"tcp.{k}"] = float
        # Gripper
        if self.config.use_gripper:
            features["gripper.pos"] = float
        # Cameras
        for cam_key in self.cameras:
            cam_cfg = self.config.cameras[cam_key]
            features[cam_key] = (cam_cfg.height, cam_cfg.width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        # EE pose action (Cartesian)
        for k in ["x", "y", "z", "wx", "wy", "wz"]:
            features[f"ee.{k}"] = float
        # Gripper position
        if self.config.use_gripper:
            features["ee.gripper_pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        ctrl_ok = self._rtde_control is not None and self._rtde_control.isConnected()
        recv_ok = self._rtde_receive is not None and self._rtde_receive.isConnected()
        cams_ok = all(cam.is_connected for cam in self.cameras.values())
        return ctrl_ok and recv_ok and cams_ok

    def connect(self, calibrate: bool = False) -> None:
        if self._rtde_control is not None:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface
        except ImportError as e:
            raise ImportError(
                "ur-rtde is required for UR5Follower. "
                "Install it with: pip install ur-rtde"
            ) from e

        print(f"DEBUG: Connecting to UR5 at {self.config.ip_address} (frequency: {self.config.rtde_frequency}Hz)...")
        self._rtde_receive = RTDEReceiveInterface(self.config.ip_address, frequency=self.config.rtde_frequency)
        print(f"DEBUG: RTDEReceiveInterface initialized.")
        time.sleep(0.2)
        print(f"DEBUG: RTDEReceiveInterface initialized. isConnected() = {self._rtde_receive.isConnected()}")

        # Connect gripper using socket implementation from user's scripts FIRST
        # This prevents RTDE timeout while gripper initializes
        if self.config.use_gripper:
            self._gripper = RobotiqGripperSocket(self.config.ip_address, 63352)
            self._gripper.connect()
            self._gripper.activate()
            self._async_gripper = AsyncGripperWrapper(self._gripper)
            self._async_gripper.start()
            logger.info("Robotiq Hand-E gripper activated via socket (async I/O)")
        
        # Keep retrying connection and give clear Fieldbus warning if it registers are in use
        last_err: Exception | None = None
        for _ in range(3):
            try:
                self._rtde_control = RTDEControlInterface(self.config.ip_address)
                time.sleep(0.2)
                if self._rtde_control.isConnected():
                    break
            except Exception as e:
                last_err = e
                if "RTDE input registers are already in use" in str(e):
                    logger.warning(
                        "RTDE control cannot start because RTDE input registers are in use.\n"
                        "On the teach pendant disable Fieldbus adapters that reserve registers:\n"
                        "- Installation -> Fieldbus -> EtherNet/IP (disable)\n"
                        "- Installation -> Fieldbus -> PROFINET (disable)\n"
                        "- Installation -> Fieldbus -> MODBUS (disable any units)\n"
                        "Then fully reboot the robot controller and retry."
                    )
                time.sleep(0.8)
        else:
            raise ConnectionError(f"Failed to connect to UR5 RTDE control at {self.config.ip_address}: {last_err}")

        logger.info(f"Connected to UR5 via RTDE at {self.config.ip_address}")

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # UR5 has factory calibration
        return True

    def calibrate(self) -> None:
        # UR5 is factory-calibrated, no additional calibration needed
        pass

    def configure(self) -> None:
        # No additional configuration needed — RTDE handles everything
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict: dict[str, Any] = {}

        # Read joint positions (radians from RTDE → degrees for consistency)
        start = time.perf_counter()
        joint_positions = self._rtde_receive.getActualQ()  # 6 values in radians
        for name, val in zip(UR5_JOINT_NAMES, joint_positions):
            obs_dict[f"{name}.pos"] = np.degrees(val)

        # Read TCP pose [x, y, z, rx, ry, rz] (meters + axis-angle radians)
        tcp_pose = self._rtde_receive.getActualTCPPose()
        for k, v in zip(["x", "y", "z", "rx", "ry", "rz"], tcp_pose):
            obs_dict[f"tcp.{k}"] = float(v)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Read gripper position (non-blocking, uses cached value from background thread)
        if self.config.use_gripper and self._async_gripper is not None:
            obs_dict["gripper.pos"] = self._async_gripper.get_cached_position()

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send a Cartesian EE pose action to the UR5 via servoL.

        Expected action keys: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz
        The rotation (wx, wy, wz) is in axis-angle (rotation vector) format.

        Optionally ee.gripper_pos (0-100) for the Robotiq Hand-E gripper.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Build the 6D pose for servoL: [x, y, z, rx, ry, rz]
        pose = [
            float(action["ee.x"]),
            float(action["ee.y"]),
            float(action["ee.z"]),
            float(action["ee.wx"]),
            float(action["ee.wy"]),
            float(action["ee.wz"]),
        ]

        # Send Cartesian pose via servoL — the UR5 controller handles IK internally
        dt = 1.0 / self.config.rtde_frequency
        self._rtde_control.servoL(
            pose,
            self.config.servo_speed,
            self.config.servo_acceleration,
            dt,
            self.config.servo_lookahead_time,
            self.config.servo_gain,
        )

        # Send gripper command
        if self.config.use_gripper and self._async_gripper is not None and "ee.gripper_pos" in action:
            # Convert 0-100 range back to 0-255 for Robotiq (non-blocking)
            gripper_target = int(float(action["ee.gripper_pos"]) / 100.0 * 255)
            gripper_target = max(0, min(255, gripper_target))
            self._async_gripper.send_command(gripper_target, self.config.gripper_speed, self.config.gripper_force)

        return action

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Stop servo mode
        if self._rtde_control is not None:
            self._rtde_control.servoStop()
            self._rtde_control.stopScript()
            self._rtde_control.disconnect()
            self._rtde_control = None

        if self._rtde_receive is not None:
            self._rtde_receive.disconnect()
            self._rtde_receive = None

        if self._async_gripper is not None:
            self._async_gripper.stop()
            self._async_gripper = None
        self._gripper = None

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
