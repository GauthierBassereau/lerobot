"""
Processor steps specific to the UR5 robot.

Since the UR5 controller handles IK internally (via servoL), these steps
replace the placo-based InverseKinematicsEEToJoints and EEReferenceAndDelta
steps used by SO100. The UR5 provides its TCP pose directly via RTDE,
so no URDF or external kinematics solver is needed.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ObservationProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    TransitionKey,
)
from lerobot.utils.rotation import Rotation


@ProcessorStepRegistry.register("ee_reference_and_delta_from_tcp")
@dataclass
class EEReferenceAndDeltaFromTCP(RobotActionProcessorStep):
    """
    Computes a target EE pose from phone delta commands using the UR5's TCP pose.

    Unlike the standard EEReferenceAndDelta which requires a kinematics solver
    for FK, this step reads the TCP pose directly from the UR5's observations.
    This eliminates the need for a URDF file or placo dependency.

    Attributes:
        end_effector_step_sizes: Scaling factors for the delta commands.
        use_latched_reference: If True, latch reference on enable; else always use current.
    """

    end_effector_step_sizes: dict
    use_latched_reference: bool = True

    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    reference_phone_rot: Rotation | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _command_when_disabled: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        if observation is None:
            raise ValueError("Observation is required for computing EE reference")

        # Read current TCP pose directly from UR5 observations (no FK needed!)
        tcp_x = float(observation.get("tcp.x", 0.0))
        tcp_y = float(observation.get("tcp.y", 0.0))
        tcp_z = float(observation.get("tcp.z", 0.0))
        tcp_rx = float(observation.get("tcp.rx", 0.0))
        tcp_ry = float(observation.get("tcp.ry", 0.0))
        tcp_rz = float(observation.get("tcp.rz", 0.0))

        # Build 4x4 transform from TCP pose
        t_curr = np.eye(4, dtype=float)
        t_curr[:3, :3] = Rotation.from_rotvec([tcp_rx, tcp_ry, tcp_rz]).as_matrix()
        t_curr[:3, 3] = [tcp_x, tcp_y, tcp_z]

        enabled = bool(action.pop("enabled"))
        tx_input = float(action.pop("target_x"))
        ty_input = float(action.pop("target_y"))
        tz_input = float(action.pop("target_z"))
        wx_input = float(action.pop("target_wx"))
        wy_input = float(action.pop("target_wy"))
        wz_input = float(action.pop("target_wz"))
        # Apply the phone axis mapping here (swapping x/z and y/x):
        tx = ty_input
        ty = -tx_input
        tz = tz_input
        wx = -wy_input
        wy = wx_input
        wz = wz_input
        gripper_vel = float(action.pop("gripper_vel"))

        desired = None

        if enabled:
            ref = t_curr
            if self.use_latched_reference:
                if not self._prev_enabled or self.reference_ee_pose is None or self.reference_phone_rot is None:
                    self.reference_ee_pose = t_curr.copy()
                    self.reference_phone_rot = Rotation.from_rotvec([wx, wy, wz])
                ref = self.reference_ee_pose if self.reference_ee_pose is not None else t_curr

                # Compute relative rotation from the latched phone orientation
                r_rel = (self.reference_phone_rot.inv() * Rotation.from_rotvec([wx, wy, wz])).as_matrix()
                desired_rot = ref[:3, :3] @ r_rel
            else:
                r_abs = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
                desired_rot = ref[:3, :3] @ r_abs

            delta_p = np.array(
                [
                    tx * self.end_effector_step_sizes["x"],
                    ty * self.end_effector_step_sizes["y"],
                    tz * self.end_effector_step_sizes["z"],
                ],
                dtype=float,
            )
            desired = np.eye(4, dtype=float)
            desired[:3, :3] = desired_rot
            desired[:3, 3] = ref[:3, 3] + delta_p

            self._command_when_disabled = desired.copy()
        else:
            if self._command_when_disabled is None:
                self._command_when_disabled = t_curr.copy()
            desired = self._command_when_disabled.copy()

        # Write action fields
        pos = desired[:3, 3]
        tw = Rotation.from_matrix(desired[:3, :3]).as_rotvec()
        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(tw[0])
        action["ee.wy"] = float(tw[1])
        action["ee.wz"] = float(tw[2])
        action["ee.gripper_vel"] = gripper_vel

        self._prev_enabled = enabled
        return action

    def reset(self):
        self._prev_enabled = False
        self.reference_ee_pose = None
        self.reference_phone_rot = None
        self._command_when_disabled = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in [
            "enabled", "target_x", "target_y", "target_z",
            "target_wx", "target_wy", "target_wz", "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION].pop(f"{feat}", None)

        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_vel"]:
            features[PipelineFeatureType.ACTION][f"ee.{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features


@ProcessorStepRegistry.register("ee_to_ur5_action")
@dataclass
class EEToUR5Action(RobotActionProcessorStep):
    """
    Passes through the end-effector pose directly as robot action for UR5.

    Unlike SO100 which needs an external IK solver, the UR5 accepts Cartesian
    commands via servoL, so this step simply passes the EE pose through.
    The gripper position is also passed through.

    Input action keys: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos
    Output action keys: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos
    (passed through unchanged — the UR5Follower.send_action reads these directly)
    """

    def action(self, action: RobotAction) -> RobotAction:
        # Validate that all required EE fields are present
        required = ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz"]
        for key in required:
            if key not in action:
                raise ValueError(f"Missing required key '{key}' in action for UR5")
        # Pass through — UR5Follower.send_action() reads ee.* keys directly
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # Features are unchanged — we keep the ee.* keys as-is
        return features


@ProcessorStepRegistry.register("ur5_observation_to_ee")
@dataclass
class UR5ObservationToEE(ObservationProcessorStep):
    """
    Converts UR5 joint and TCP observations to EE-space representation.

    The UR5 already provides TCP pose in its observations (tcp.x, tcp.y, etc.),
    so this step renames them to the standard ee.* format and removes joint
    positions, keeping the observation in EE space for dataset consistency.
    """

    keep_joints: bool = False

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        # Map tcp.* → ee.* and rename rotation keys
        tcp_to_ee = {
            "tcp.x": "ee.x",
            "tcp.y": "ee.y",
            "tcp.z": "ee.z",
            "tcp.rx": "ee.wx",
            "tcp.ry": "ee.wy",
            "tcp.rz": "ee.wz",
        }
        for tcp_key, ee_key in tcp_to_ee.items():
            if tcp_key in observation:
                observation[ee_key] = observation.pop(tcp_key)

        # Optionally remove joint observations (keep only EE space)
        if not self.keep_joints:
            joint_keys = [k for k in observation if k.endswith(".pos") and k != "gripper.pos"]
            for k in joint_keys:
                observation.pop(k)

        # Rename gripper.pos → ee.gripper_pos for consistency
        if "gripper.pos" in observation:
            observation["ee.gripper_pos"] = observation.pop("gripper.pos")

        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        obs_features = features[PipelineFeatureType.OBSERVATION]

        # Remove tcp.* and add ee.*
        for tcp_key in ["tcp.x", "tcp.y", "tcp.z", "tcp.rx", "tcp.ry", "tcp.rz"]:
            obs_features.pop(tcp_key, None)

        for ee_key in ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz"]:
            obs_features[ee_key] = PolicyFeature(type=FeatureType.STATE, shape=(1,))

        # Remove joint features if not keeping them
        if not self.keep_joints:
            joint_keys = [k for k in list(obs_features.keys()) if k.endswith(".pos") and k != "gripper.pos"]
            for k in joint_keys:
                obs_features.pop(k, None)

        # Rename gripper
        if "gripper.pos" in obs_features:
            obs_features.pop("gripper.pos")
            obs_features["ee.gripper_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(1,))

        return features
