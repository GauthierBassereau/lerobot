#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add EE-relative action features to a LeRobot dataset.

This script keeps the original absolute `action` feature and adds two derived features:

1. `action.delta_base`
   Target EE delta relative to the same-frame EE observation, expressed in the robot/base frame.

2. `action.delta_local`
   Target EE delta relative to the same-frame EE observation, expressed in the current EE frame.

Both features use the current frame's `observation.state` EE pose as the reference pose:

    delta_pos_base  = p_target - p_obs
    delta_pos_local = R_obs^T (p_target - p_obs)

    R_delta_base  = R_target R_obs^T
    R_delta_local = R_obs^T R_target

The orientation terms are stored as rotation vectors, matching the existing UR5 `ee.wx/wy/wz`
representation used in the dataset.

Example:
    python -m lerobot.scripts.add_relative_ee_action_features \
        --repo-id Gaugou/ur5 \
        --output-repo-id Gaugou/ur5_with_action_deltas
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.io_utils import load_episodes, write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import flatten_dict
from lerobot.utils.rotation import Rotation

ACTION_FEATURE = "action"
OBSERVATION_FEATURE = "observation.state"
BASE_DELTA_FEATURE = "action.delta_base"
LOCAL_DELTA_FEATURE = "action.delta_local"

EE_POSE_NAMES = (
    "ee.x",
    "ee.y",
    "ee.z",
    "ee.wx",
    "ee.wy",
    "ee.wz",
    "ee.gripper_pos",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Input LeRobot dataset repo id.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional local dataset root containing meta/, data/, and videos/.",
    )
    parser.add_argument(
        "--output-repo-id",
        default=None,
        help="Output dataset repo id. Defaults to '<repo-id>_with_action_deltas'.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output dataset root. Defaults to the standard LeRobot cache for output repo id.",
    )
    parser.add_argument(
        "--action-feature",
        default=ACTION_FEATURE,
        help=f"Feature key holding the absolute action vector. Default: {ACTION_FEATURE}",
    )
    parser.add_argument(
        "--observation-feature",
        default=OBSERVATION_FEATURE,
        help=(
            "Feature key holding the EE observation vector used as the reference pose. "
            f"Default: {OBSERVATION_FEATURE}"
        ),
    )
    parser.add_argument(
        "--base-feature-name",
        default=BASE_DELTA_FEATURE,
        help=f"Name of the added base-frame delta feature. Default: {BASE_DELTA_FEATURE}",
    )
    parser.add_argument(
        "--local-feature-name",
        default=LOCAL_DELTA_FEATURE,
        help=f"Name of the added EE-local delta feature. Default: {LOCAL_DELTA_FEATURE}",
    )
    parser.add_argument(
        "--overwrite-stats-only",
        action="store_true",
        help=(
            "Skip dataset copying and only recompute stats/episode stats for the already-created output dataset. "
            "Useful if the previous run finished copying data but was interrupted before metadata updates."
        ),
    )
    return parser.parse_args()


def get_feature_indices(feature_key: str, feature_info: dict, required_names: tuple[str, ...]) -> dict[str, int]:
    feature_names = feature_info.get("names")
    if feature_names is None:
        raise ValueError(
            f"Feature '{feature_key}' must define a 'names' list so the EE components can be located."
        )

    missing = [name for name in required_names if name not in feature_names]
    if missing:
        raise ValueError(
            f"Feature '{feature_key}' is missing required EE entries: {missing}. "
            f"Available names: {feature_names}"
        )

    return {name: feature_names.index(name) for name in required_names}


def extract_pose(vector: np.ndarray, indices: dict[str, int]) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(vector, dtype=np.float64)
    position = values[[indices["ee.x"], indices["ee.y"], indices["ee.z"]]]
    rotvec = values[[indices["ee.wx"], indices["ee.wy"], indices["ee.wz"]]]
    gripper = float(values[indices["ee.gripper_pos"]])
    return position, rotvec, gripper


def compute_action_deltas(
    action_vector: np.ndarray,
    observation_vector: np.ndarray,
    action_indices: dict[str, int],
    observation_indices: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    action_position, action_rotvec, action_gripper = extract_pose(action_vector, action_indices)
    obs_position, obs_rotvec, obs_gripper = extract_pose(observation_vector, observation_indices)

    obs_rotation = Rotation.from_rotvec(obs_rotvec)
    action_rotation = Rotation.from_rotvec(action_rotvec)

    delta_position_base = action_position - obs_position
    delta_position_local = obs_rotation.apply(delta_position_base, inverse=True)

    delta_rotation_base = (action_rotation * obs_rotation.inv()).as_rotvec()
    delta_rotation_local = (obs_rotation.inv() * action_rotation).as_rotvec()

    delta_gripper = np.array([action_gripper - obs_gripper], dtype=np.float64)

    base_delta = np.concatenate([delta_position_base, delta_rotation_base, delta_gripper]).astype(np.float32)
    local_delta = np.concatenate([delta_position_local, delta_rotation_local, delta_gripper]).astype(np.float32)

    return base_delta, local_delta


def compute_delta_arrays(
    dataset: LeRobotDataset,
    action_feature: str,
    observation_feature: str,
) -> tuple[np.ndarray, np.ndarray]:
    if action_feature not in dataset.meta.features:
        raise ValueError(f"Feature '{action_feature}' not found in dataset.")
    if observation_feature not in dataset.meta.features:
        raise ValueError(f"Feature '{observation_feature}' not found in dataset.")

    action_indices = get_feature_indices(action_feature, dataset.meta.features[action_feature], EE_POSE_NAMES)
    observation_indices = get_feature_indices(
        observation_feature, dataset.meta.features[observation_feature], EE_POSE_NAMES
    )

    parquet_files = sorted((dataset.root / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found under {dataset.root / 'data'}")

    total_frames = dataset.meta.total_frames
    base_deltas = np.empty((total_frames, len(EE_POSE_NAMES)), dtype=np.float32)
    local_deltas = np.empty((total_frames, len(EE_POSE_NAMES)), dtype=np.float32)

    offset = 0
    progress = tqdm(parquet_files, desc="Computing action deltas")
    for parquet_path in progress:
        df = pd.read_parquet(parquet_path, columns=[action_feature, observation_feature])
        action_values = np.stack(df[action_feature].to_list()).astype(np.float64)
        observation_values = np.stack(df[observation_feature].to_list()).astype(np.float64)

        num_rows = len(df)
        for row_idx in range(num_rows):
            base_delta, local_delta = compute_action_deltas(
                action_values[row_idx],
                observation_values[row_idx],
                action_indices,
                observation_indices,
            )
            base_deltas[offset + row_idx] = base_delta
            local_deltas[offset + row_idx] = local_delta

        offset += num_rows

    if offset != total_frames:
        raise ValueError(f"Expected {total_frames} frames but computed {offset} rows of deltas.")

    return base_deltas, local_deltas


def build_feature_info(suffix: str) -> dict:
    return {
        "dtype": "float32",
        "shape": [7],
        "names": [
            f"ee.dx_{suffix}",
            f"ee.dy_{suffix}",
            f"ee.dz_{suffix}",
            f"ee.dwx_{suffix}",
            f"ee.dwy_{suffix}",
            f"ee.dwz_{suffix}",
            "ee.dgripper_pos",
        ],
    }


def compute_feature_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    return get_feature_stats(values, axis=0, keepdims=False)


def to_object_feature_column(values: np.ndarray) -> np.ndarray:
    """Convert a dense [N, D] array into a 1D object array of per-row vectors.

    `lerobot.datasets.dataset_tools.add_features()` currently writes tabular features through
    a pandas DataFrame assignment path that expects a 1D column. For vector-valued features we
    therefore provide a column of `np.ndarray` rows instead of a single 2D array.
    """
    object_column = np.empty((len(values),), dtype=object)
    for idx, row in enumerate(values):
        object_column[idx] = row
    return object_column


def compute_episode_feature_stats(
    dataset: LeRobotDataset,
    feature_arrays: dict[str, np.ndarray],
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.root)

    episode_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for episode_idx in tqdm(range(dataset.meta.total_episodes), desc="Computing episode stats"):
        start = int(dataset.meta.episodes["dataset_from_index"][episode_idx])
        end = int(dataset.meta.episodes["dataset_to_index"][episode_idx])
        episode_stats[episode_idx] = {
            feature_name: compute_feature_stats(feature_values[start:end])
            for feature_name, feature_values in feature_arrays.items()
        }
    return episode_stats


def update_episode_stats_files(
    output_root: Path,
    episode_feature_stats: dict[int, dict[str, dict[str, np.ndarray]]],
) -> None:
    episode_files = sorted((output_root / "meta" / "episodes").glob("*/*.parquet"))
    if not episode_files:
        raise ValueError(f"No episode metadata files found under {output_root / 'meta' / 'episodes'}")

    for episode_file in tqdm(episode_files, desc="Updating episode metadata"):
        df = pd.read_parquet(episode_file)
        extra_columns: dict[str, list[np.ndarray]] = {}

        for episode_idx in df["episode_index"].astype(int).tolist():
            flat_stats = flatten_dict({"stats": episode_feature_stats[episode_idx]})
            for column, value in flat_stats.items():
                extra_columns.setdefault(column, []).append(value)

        for column, values in extra_columns.items():
            df[column] = values

        df.to_parquet(episode_file, index=False)


def update_dataset_stats_file(
    dataset: LeRobotDataset,
    feature_arrays: dict[str, np.ndarray],
) -> None:
    updated_stats = {}
    if dataset.meta.stats is not None:
        updated_stats.update(dataset.meta.stats)

    for feature_name, values in feature_arrays.items():
        updated_stats[feature_name] = compute_feature_stats(values)

    write_stats(updated_stats, dataset.root)


def add_or_update_features(args: argparse.Namespace) -> None:
    output_repo_id = args.output_repo_id or f"{args.repo_id}_with_action_deltas"
    output_root = args.output_root

    src_dataset = LeRobotDataset(args.repo_id, root=args.root)
    base_deltas, local_deltas = compute_delta_arrays(
        src_dataset,
        action_feature=args.action_feature,
        observation_feature=args.observation_feature,
    )

    feature_arrays = {
        args.base_feature_name: base_deltas,
        args.local_feature_name: local_deltas,
    }

    if not args.overwrite_stats_only:
        logging.info("Creating dataset copy with new action-delta features")
        add_features(
            dataset=src_dataset,
            features={
                args.base_feature_name: (to_object_feature_column(base_deltas), build_feature_info("base")),
                args.local_feature_name: (to_object_feature_column(local_deltas), build_feature_info("local")),
            },
            output_dir=output_root,
            repo_id=output_repo_id,
        )

    dst_dataset = LeRobotDataset(output_repo_id, root=output_root)
    episode_feature_stats = compute_episode_feature_stats(src_dataset, feature_arrays)
    update_episode_stats_files(dst_dataset.root, episode_feature_stats)
    update_dataset_stats_file(dst_dataset, feature_arrays)

    logging.info("Done")
    logging.info("Output dataset: %s", dst_dataset.root)
    logging.info("Added features: %s, %s", args.base_feature_name, args.local_feature_name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    add_or_update_features(args)


if __name__ == "__main__":
    main()
