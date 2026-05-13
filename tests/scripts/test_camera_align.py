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

import numpy as np

from lerobot.scripts.lerobot_camera_align import (
    blend_reference_overlay,
    fit_reference_to_frame,
    overlay_reference_edges,
)


def test_fit_reference_to_frame_stretch_uses_full_frame():
    reference = np.full((2, 4, 3), 25, dtype=np.uint8)

    fitted_reference, valid_mask = fit_reference_to_frame(reference, (6, 8, 3), fit_mode="stretch")

    assert fitted_reference.shape == (6, 8, 3)
    assert valid_mask.shape == (6, 8)
    assert valid_mask.all()


def test_fit_reference_to_frame_contain_preserves_aspect_ratio():
    reference = np.full((2, 4, 3), 255, dtype=np.uint8)

    fitted_reference, valid_mask = fit_reference_to_frame(reference, (8, 8, 3), fit_mode="contain")

    assert fitted_reference.shape == (8, 8, 3)
    assert valid_mask.shape == (8, 8)
    assert valid_mask.sum() == 32
    assert not valid_mask[:2].any()
    assert valid_mask[2:6].all()
    assert not valid_mask[6:].any()


def test_blend_reference_overlay_respects_mask():
    live_frame = np.full((2, 2, 3), 100, dtype=np.uint8)
    reference_frame = np.full((2, 2, 3), 200, dtype=np.uint8)
    valid_mask = np.array([[True, False], [False, True]])

    blended = blend_reference_overlay(live_frame, reference_frame, valid_mask, alpha=0.25)

    assert blended[0, 0].tolist() == [125, 125, 125]
    assert blended[1, 1].tolist() == [125, 125, 125]
    assert blended[0, 1].tolist() == [100, 100, 100]
    assert blended[1, 0].tolist() == [100, 100, 100]


def test_overlay_reference_edges_only_updates_edge_pixels():
    live_frame = np.zeros((3, 3, 3), dtype=np.uint8)
    reference_edges = np.zeros((3, 3), dtype=np.uint8)
    valid_mask = np.zeros((3, 3), dtype=bool)
    reference_edges[1, 1] = 255
    valid_mask[1, 1] = True

    output = overlay_reference_edges(live_frame, reference_edges, valid_mask)

    assert output[1, 1].tolist() == [0, 255, 255]
    assert output.sum() == 510
