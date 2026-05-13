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

This script always uses a future EE observation as the target pose and the
current frame's `observation.state` as the reference pose.

- Default behavior: if neither `--target-fps` nor `--future-offset-frames` is
  passed, use the next frame as the target (`t -> t+1`). This corresponds to
  the dataset's original FPS.
- Coarse behavior: if `--target-fps` is provided, use
  `stride = dataset_fps / target_fps`. For a 30 Hz dataset and
  `--target-fps 5`, the stride is 6, so zero-based `frame 0 -> frame 6` and
  one-based `t1 -> t7`.
- Explicit behavior: `--future-offset-frames N` uses `observation_{t+N}` as the
  target directly.

If the requested future frame falls past the end of an episode, the target is
clamped to that episode's last frame. This matches LeRobot's end-padding
behavior when requesting out-of-range future frames.

The computed deltas are:

    delta_pos_base  = p_target - p_obs
    delta_pos_local = R_obs^T (p_target - p_obs)

    R_delta_base  = R_target R_obs^T
    R_delta_local = R_obs^T R_target

The orientation terms are stored as rotation vectors, matching the existing UR5
`ee.wx/wy/wz` representation used in the dataset.

Example:
    python -m lerobot.scripts.add_relative_ee_action_features \
        --repo-id Gaugou/ur5_wild \
        --target-fps 5 \
        --output-repo-id Gaugou/ur5_wild_5hz
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
from lerobot.datasets.io_utils import write_stats
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
        help=(
            "Deprecated and ignored. Targets are always derived from future observations. "
            f"Retained only for CLI compatibility; default: {ACTION_FEATURE}"
        ),
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
        default=None,
        help=(
            "Name of the added base-frame delta feature. Defaults to "
            f"'{BASE_DELTA_FEATURE}' for the default stride-1 future target, or "
            "'action.<target-fps>hz_delta_base' when --target-fps is provided."
        ),
    )
    parser.add_argument(
        "--local-feature-name",
        default=None,
        help=(
            "Name of the added EE-local delta feature. Defaults to "
            f"'{LOCAL_DELTA_FEATURE}' for the default stride-1 future target, or "
            "'action.<target-fps>hz_delta_local' when --target-fps is provided."
        ),
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help=(
            "Optional lower target FPS for coarse endpoint deltas. When omitted, the script uses "
            "the dataset's original FPS and computes deltas from observation_t to observation_{t+1}. "
            "When set, it computes deltas from observation_t to observation_{t+stride}, where "
            "stride = dataset_fps / target_fps."
        ),
    )
    parser.add_argument(
        "--future-offset-frames",
        type=int,
        default=None,
        help=(
            "Optional explicit future offset in frames. Mutually exclusive with --target-fps. "
            "Use this to define the future target directly as observation_{t+offset}. "
            "When neither this nor --target-fps is provided, the default offset is 1 frame."
        ),
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


def format_fps_tag(fps: float) -> str:
    return format(fps, "g")


def infer_output_feature_names(
    base_feature_name: str | None,
    local_feature_name: str | None,
    target_fps: float | None,
    future_offset_frames: int | None = None,
) -> tuple[str, str]:
    if target_fps is None and future_offset_frames is None:
        default_base = BASE_DELTA_FEATURE
        default_local = LOCAL_DELTA_FEATURE
    elif target_fps is not None:
        fps_tag = format_fps_tag(target_fps)
        default_base = f"action.{fps_tag}hz_delta_base"
        default_local = f"action.{fps_tag}hz_delta_local"
    else:
        default_base = f"action.offset{future_offset_frames}_delta_base"
        default_local = f"action.offset{future_offset_frames}_delta_local"

    return base_feature_name or default_base, local_feature_name or default_local


def resolve_future_frame_stride(
    dataset_fps: int,
    target_fps: float | None,
    future_offset_frames: int | None,
) -> int:
    if target_fps is not None and future_offset_frames is not None:
        raise ValueError("Use either --target-fps or --future-offset-frames, not both.")

    if future_offset_frames is not None:
        if future_offset_frames <= 0:
            raise ValueError(f"--future-offset-frames must be > 0, got {future_offset_frames}.")
        return future_offset_frames

    if target_fps is None:
        return 1

    if target_fps <= 0:
        raise ValueError(f"--target-fps must be > 0, got {target_fps}.")
    if target_fps > dataset_fps:
        raise ValueError(
            f"--target-fps ({target_fps}) cannot exceed dataset fps ({dataset_fps}) for this downsampling workflow."
        )

    stride = dataset_fps / target_fps
    rounded_stride = round(stride)
    if not np.isclose(stride, rounded_stride, atol=1e-6):
        raise ValueError(
            f"Dataset fps {dataset_fps} is not an integer multiple of target fps {target_fps}. "
            "Pass --future-offset-frames explicitly if you want a non-integer ratio."
        )

    return int(rounded_stride)


def compute_pose_deltas(
    target_vector: np.ndarray,
    reference_vector: np.ndarray,
    target_indices: dict[str, int],
    reference_indices: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    target_position, target_rotvec, target_gripper = extract_pose(target_vector, target_indices)
    reference_position, reference_rotvec, reference_gripper = extract_pose(reference_vector, reference_indices)

    reference_rotation = Rotation.from_rotvec(reference_rotvec)
    target_rotation = Rotation.from_rotvec(target_rotvec)

    delta_position_base = target_position - reference_position
    delta_position_local = reference_rotation.apply(delta_position_base, inverse=True)

    delta_rotation_base = (target_rotation * reference_rotation.inv()).as_rotvec()
    delta_rotation_local = (reference_rotation.inv() * target_rotation).as_rotvec()

    delta_gripper = np.array([target_gripper - reference_gripper], dtype=np.float64)

    base_delta = np.concatenate([delta_position_base, delta_rotation_base, delta_gripper]).astype(np.float32)
    local_delta = np.concatenate([delta_position_local, delta_rotation_local, delta_gripper]).astype(np.float32)

    return base_delta, local_delta


def load_feature_values(dataset: LeRobotDataset, feature_key: str) -> np.ndarray:
    if feature_key not in dataset.meta.features:
        raise ValueError(f"Feature '{feature_key}' not found in dataset.")

    parquet_files = sorted((dataset.root / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found under {dataset.root / 'data'}")

    values: np.ndarray | None = None
    offset = 0
    progress = tqdm(parquet_files, desc=f"Loading {feature_key}")
    for parquet_path in progress:
        df = pd.read_parquet(parquet_path, columns=[feature_key])
        chunk_values = np.stack(df[feature_key].to_list()).astype(np.float64)
        if values is None:
            values = np.empty((dataset.meta.total_frames, chunk_values.shape[1]), dtype=np.float64)
        values[offset : offset + len(df)] = chunk_values
        offset += len(df)

    if values is None:
        raise ValueError(f"Could not load any values for feature '{feature_key}'.")
    if offset != dataset.meta.total_frames:
        raise ValueError(f"Expected {dataset.meta.total_frames} frames but loaded {offset} rows for {feature_key}.")

    return values


def compute_future_observation_delta_arrays(
    dataset: LeRobotDataset,
    observation_feature: str,
    future_offset_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if future_offset_frames <= 0:
        raise ValueError(f"future_offset_frames must be > 0, got {future_offset_frames}.")

    observation_indices = get_feature_indices(
        observation_feature, dataset.meta.features[observation_feature], EE_POSE_NAMES
    )
    observation_values = load_feature_values(dataset, observation_feature)
    episodes = ensure_episode_metadata_loaded(dataset)

    total_frames = dataset.meta.total_frames
    base_deltas = np.empty((total_frames, len(EE_POSE_NAMES)), dtype=np.float32)
    local_deltas = np.empty((total_frames, len(EE_POSE_NAMES)), dtype=np.float32)

    clamped_frames = 0
    for episode_idx in tqdm(range(dataset.meta.total_episodes), desc="Computing future observation deltas"):
        start = int(episodes["dataset_from_index"][episode_idx])
        end = int(episodes["dataset_to_index"][episode_idx])
        for abs_idx in range(start, end):
            target_idx = abs_idx + future_offset_frames
            if target_idx >= end:
                target_idx = end - 1
                clamped_frames += 1

            base_delta, local_delta = compute_pose_deltas(
                observation_values[target_idx],
                observation_values[abs_idx],
                observation_indices,
                observation_indices,
            )
            base_deltas[abs_idx] = base_delta
            local_deltas[abs_idx] = local_delta

    if clamped_frames > 0:
        logging.info(
            "Clamped %d trailing frames to the episode end because they do not have a full future horizon of %d frames.",
            clamped_frames,
            future_offset_frames,
        )

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


def load_episode_metadata_local(root: Path) -> pd.DataFrame:
    episode_files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not episode_files:
        raise ValueError(f"No episode metadata files found under {root / 'meta' / 'episodes'}")

    episodes = pd.concat((pd.read_parquet(path) for path in episode_files), ignore_index=True)
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    return episodes


def ensure_episode_metadata_loaded(dataset: LeRobotDataset) -> pd.DataFrame:
    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episode_metadata_local(dataset.root)
    return dataset.meta.episodes


def compute_episode_feature_stats(
    dataset: LeRobotDataset,
    feature_arrays: dict[str, np.ndarray],
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    episodes = ensure_episode_metadata_loaded(dataset)

    episode_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for episode_idx in tqdm(range(dataset.meta.total_episodes), desc="Computing episode stats"):
        start = int(episodes["dataset_from_index"][episode_idx])
        end = int(episodes["dataset_to_index"][episode_idx])
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
    if args.action_feature != ACTION_FEATURE:
        logging.warning(
            "Ignoring --action-feature=%r. Targets are now always computed from future observations.",
            args.action_feature,
        )
    future_offset_frames = resolve_future_frame_stride(
        dataset_fps=src_dataset.fps,
        target_fps=args.target_fps,
        future_offset_frames=args.future_offset_frames,
    )
    base_feature_name, local_feature_name = infer_output_feature_names(
        base_feature_name=args.base_feature_name,
        local_feature_name=args.local_feature_name,
        target_fps=args.target_fps,
        future_offset_frames=args.future_offset_frames,
    )

    base_deltas, local_deltas = compute_future_observation_delta_arrays(
        src_dataset,
        observation_feature=args.observation_feature,
        future_offset_frames=future_offset_frames,
    )
    if args.future_offset_frames is not None:
        logging.info(
            "Using future observation targets with an explicit stride of %d frames.",
            future_offset_frames,
        )
    elif args.target_fps is None:
        logging.info(
            "Using future observation targets with the default stride of 1 frame at the dataset FPS (%s Hz).",
            src_dataset.fps,
        )
    else:
        logging.info(
            "Using future observation targets with a stride of %d frames (%s Hz dataset -> %s Hz target).",
            future_offset_frames,
            src_dataset.fps,
            format_fps_tag(args.target_fps) if args.target_fps is not None else f"+{future_offset_frames} frames",
        )

    feature_arrays = {
        base_feature_name: base_deltas,
        local_feature_name: local_deltas,
    }

    if not args.overwrite_stats_only:
        logging.info("Creating dataset copy with new action-delta features")
        add_features(
            dataset=src_dataset,
            features={
                base_feature_name: (to_object_feature_column(base_deltas), build_feature_info("base")),
                local_feature_name: (to_object_feature_column(local_deltas), build_feature_info("local")),
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
    logging.info("Added features: %s, %s", base_feature_name, local_feature_name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    add_or_update_features(args)


if __name__ == "__main__":
    main()
