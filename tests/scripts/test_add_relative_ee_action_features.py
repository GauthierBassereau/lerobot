#!/usr/bin/env python

import numpy as np
import torch

from lerobot.scripts.add_relative_ee_action_features import (
    compute_action_deltas,
    compute_future_observation_delta_arrays,
    infer_output_feature_names,
    resolve_future_frame_stride,
    to_object_feature_column,
)
from lerobot.utils.rotation import Rotation


INDICES = {
    "ee.x": 0,
    "ee.y": 1,
    "ee.z": 2,
    "ee.wx": 3,
    "ee.wy": 4,
    "ee.wz": 5,
    "ee.gripper_pos": 6,
}


def test_compute_action_deltas_identity_reference():
    observation = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0], dtype=np.float64)
    action = np.array([1.0, 2.0, 3.0, 0.0, 0.0, np.pi / 2.0, 25.0], dtype=np.float64)

    base_delta, local_delta = compute_action_deltas(action, observation, INDICES, INDICES)

    expected = np.array([1.0, 2.0, 3.0, 0.0, 0.0, np.pi / 2.0, 15.0], dtype=np.float32)
    np.testing.assert_allclose(base_delta, expected, atol=1e-6)
    np.testing.assert_allclose(local_delta, expected, atol=1e-6)


def test_compute_action_deltas_local_frame_translation_and_rotation():
    obs_rotvec = np.array([0.0, 0.0, np.pi / 2.0], dtype=np.float64)
    observation = np.array([0.0, 0.0, 0.0, *obs_rotvec, 40.0], dtype=np.float64)

    local_rotation_step = Rotation.from_rotvec([0.1, 0.0, 0.0])
    action_rotation = Rotation.from_rotvec(obs_rotvec) * local_rotation_step
    action_rotvec = action_rotation.as_rotvec()
    action = np.array([1.0, 0.0, 0.0, *action_rotvec, 55.0], dtype=np.float64)

    base_delta, local_delta = compute_action_deltas(action, observation, INDICES, INDICES)

    np.testing.assert_allclose(base_delta[:3], np.array([1.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(local_delta[:3], np.array([0.0, -1.0, 0.0], dtype=np.float32), atol=1e-6)

    np.testing.assert_allclose(local_delta[3:6], np.array([0.1, 0.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(base_delta[3:6], np.array([0.0, 0.1, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(base_delta[6], np.float32(15.0), atol=1e-6)
    np.testing.assert_allclose(local_delta[6], np.float32(15.0), atol=1e-6)


def test_to_object_feature_column_converts_rows_to_1d_object_array():
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    object_column = to_object_feature_column(values)

    assert object_column.shape == (2,)
    assert object_column.dtype == object
    np.testing.assert_allclose(object_column[0], np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(object_column[1], np.array([3.0, 4.0], dtype=np.float32))


def test_resolve_future_frame_stride_30hz_to_5hz():
    assert resolve_future_frame_stride(dataset_fps=30, target_fps=5.0, future_offset_frames=None) == 6


def test_infer_output_feature_names_for_target_fps():
    base_name, local_name = infer_output_feature_names(None, None, 5.0)
    assert base_name == "action.5hz_delta_base"
    assert local_name == "action.5hz_delta_local"


def test_compute_future_observation_delta_arrays_uses_t_plus_6_endpoint(
    tmp_path, empty_lerobot_dataset_factory
):
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"],
        }
    }

    dataset = empty_lerobot_dataset_factory(root=tmp_path / "test", features=features, use_videos=False, fps=30)

    for frame_idx in range(10):
        dataset.add_frame(
            {
                "observation.state": torch.tensor(
                    [frame_idx, 0.0, 0.0, 0.0, 0.0, 0.0, frame_idx],
                    dtype=torch.float32,
                ),
                "task": "task",
            }
        )
    dataset.save_episode()
    dataset.finalize()

    base_deltas, local_deltas = compute_future_observation_delta_arrays(
        dataset,
        observation_feature="observation.state",
        future_offset_frames=6,
    )

    # For a 30 Hz dataset and 5 Hz target, frame 0 should target frame 6.
    expected_delta = np.array([6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0], dtype=np.float32)
    np.testing.assert_allclose(base_deltas[0], expected_delta, atol=1e-6)
    np.testing.assert_allclose(local_deltas[0], expected_delta, atol=1e-6)

    # Trailing frames are clamped to the episode end when a full horizon is unavailable.
    expected_clamped_delta = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(base_deltas[6], expected_clamped_delta, atol=1e-6)
