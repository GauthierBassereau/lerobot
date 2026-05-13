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
Interactively align a live camera feed against a reference image.

This is useful when you want to reproduce the same camera framing across multiple
dataset recording sessions.

Example:

```shell
lerobot-camera-align path/to/reference.png --camera 0
```
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

WINDOW_NAME = "LeRobot Camera Align"
ALPHA_TRACKBAR = "Reference opacity (%)"
DEFAULT_ALPHA_PERCENT = 50
EDGE_COLOR = (0, 255, 255)


@dataclass
class ViewerState:
    fit_mode: str
    show_edges: bool
    mirror: bool


def parse_camera_index_or_path(value: str) -> int | Path:
    """Convert a CLI camera argument to either an integer index or a filesystem path."""
    return int(value) if value.lstrip("-").isdigit() else Path(value)


def load_reference_image(reference_image_path: Path) -> np.ndarray:
    """Load a reference image in BGR format."""
    reference_image = cv2.imread(str(reference_image_path), cv2.IMREAD_COLOR)
    if reference_image is None:
        raise FileNotFoundError(f"Failed to load reference image from: {reference_image_path}")
    return reference_image


def fit_reference_to_frame(
    reference_image: np.ndarray,
    frame_shape: tuple[int, ...],
    fit_mode: str,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize a reference image to match a frame and return a validity mask."""
    frame_height, frame_width = frame_shape[:2]

    if fit_mode == "stretch":
        fitted_reference = cv2.resize(reference_image, (frame_width, frame_height), interpolation=interpolation)
        valid_mask = np.ones((frame_height, frame_width), dtype=bool)
        return fitted_reference, valid_mask

    if fit_mode != "contain":
        raise ValueError(f"Unsupported fit mode: {fit_mode}")

    ref_height, ref_width = reference_image.shape[:2]
    scale = min(frame_width / ref_width, frame_height / ref_height)
    fitted_width = max(1, int(round(ref_width * scale)))
    fitted_height = max(1, int(round(ref_height * scale)))
    resized_reference = cv2.resize(reference_image, (fitted_width, fitted_height), interpolation=interpolation)

    if reference_image.ndim == 2:
        fitted_reference = np.zeros((frame_height, frame_width), dtype=reference_image.dtype)
    else:
        fitted_reference = np.zeros(
            (frame_height, frame_width, reference_image.shape[2]),
            dtype=reference_image.dtype,
        )

    top = (frame_height - fitted_height) // 2
    left = (frame_width - fitted_width) // 2
    bottom = top + fitted_height
    right = left + fitted_width

    fitted_reference[top:bottom, left:right] = resized_reference
    valid_mask = np.zeros((frame_height, frame_width), dtype=bool)
    valid_mask[top:bottom, left:right] = True

    return fitted_reference, valid_mask


def blend_reference_overlay(
    live_frame: np.ndarray,
    reference_frame: np.ndarray,
    valid_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend the reference image over the live frame using the requested opacity."""
    if live_frame.shape != reference_frame.shape:
        raise ValueError(
            f"Live frame shape {live_frame.shape} does not match reference frame shape {reference_frame.shape}."
        )
    if live_frame.shape[:2] != valid_mask.shape:
        raise ValueError(f"Mask shape {valid_mask.shape} does not match frame shape {live_frame.shape[:2]}.")

    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha == 0.0 or not valid_mask.any():
        return live_frame.copy()

    alpha_mask = np.where(valid_mask[..., None], alpha, 0.0).astype(np.float32)
    blended = (
        live_frame.astype(np.float32) * (1.0 - alpha_mask)
        + reference_frame.astype(np.float32) * alpha_mask
    )
    return np.round(blended).astype(np.uint8)


def compute_reference_edges(reference_image: np.ndarray, threshold1: int, threshold2: int) -> np.ndarray:
    """Compute an edge map for the reference image."""
    grayscale_reference = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(grayscale_reference, threshold1, threshold2)


def overlay_reference_edges(
    live_frame: np.ndarray,
    reference_edges: np.ndarray,
    valid_mask: np.ndarray,
    edge_color: tuple[int, int, int] = EDGE_COLOR,
) -> np.ndarray:
    """Highlight reference edges on top of the live frame."""
    if live_frame.shape[:2] != reference_edges.shape:
        raise ValueError(
            f"Reference edge shape {reference_edges.shape} does not match frame shape {live_frame.shape[:2]}."
        )
    result = live_frame.copy()
    result[(reference_edges > 0) & valid_mask] = edge_color
    return result


def draw_status_panel(
    image: np.ndarray,
    opacity_percent: int,
    viewer_state: ViewerState,
) -> np.ndarray:
    """Draw the current viewer status and controls on top of the image."""
    lines = [
        f"Opacity: {opacity_percent}%",
        f"Fit: {viewer_state.fit_mode}",
        f"Edges: {'on' if viewer_state.show_edges else 'off'}",
        f"Mirror: {'on' if viewer_state.mirror else 'off'}",
        "Keys: [e] edges  [f] fit  [m] mirror  [q] quit",
    ]

    output = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 22
    panel_padding = 10

    max_width = 0
    for line in lines:
        (line_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, line_width)

    panel_width = max_width + 2 * panel_padding
    panel_height = len(lines) * line_height + panel_padding
    cv2.rectangle(output, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), thickness=-1)

    for idx, line in enumerate(lines):
        baseline_y = 10 + panel_padding + (idx + 1) * line_height - 6
        cv2.putText(
            output,
            line,
            (20, baseline_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return output


def _noop(_: int) -> None:
    """Trackbar callback placeholder."""


def _create_window() -> None:
    """Create the HighGUI window and opacity slider."""
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(ALPHA_TRACKBAR, WINDOW_NAME, DEFAULT_ALPHA_PERCENT, 100, _noop)
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV GUI support is not available. Install a GUI-enabled OpenCV build such as `opencv-python`."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("reference_image", type=Path, help="Path to the reference image used for alignment.")
    parser.add_argument(
        "--camera",
        default="0",
        help="OpenCV camera index or device path. Defaults to the first webcam (`0`).",
    )
    parser.add_argument("--width", type=int, default=None, help="Requested camera width in pixels.")
    parser.add_argument("--height", type=int, default=None, help="Requested camera height in pixels.")
    parser.add_argument("--fps", type=int, default=None, help="Requested camera FPS.")
    parser.add_argument("--warmup-s", type=int, default=1, help="Warmup time in seconds before showing frames.")
    parser.add_argument(
        "--fit-mode",
        choices=("stretch", "contain"),
        default="stretch",
        help="How to fit the reference image into the live camera frame.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the preview horizontally. This mirrors both the live feed and the reference overlay.",
    )
    parser.add_argument(
        "--edge-threshold1",
        type=int,
        default=80,
        help="Low threshold for reference edge detection.",
    )
    parser.add_argument(
        "--edge-threshold2",
        type=int,
        default=160,
        help="High threshold for reference edge detection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_image = load_reference_image(args.reference_image)
    camera_index_or_path = parse_camera_index_or_path(args.camera)
    viewer_state = ViewerState(fit_mode=args.fit_mode, show_edges=False, mirror=args.mirror)

    camera_config = OpenCVCameraConfig(
        index_or_path=camera_index_or_path,
        color_mode=ColorMode.BGR,
        fps=args.fps,
        width=args.width,
        height=args.height,
        warmup_s=args.warmup_s,
    )

    _create_window()
    print(f"Reference image: {args.reference_image}")
    print("Controls: [e] toggle edges, [f] toggle fit mode, [m] toggle mirror, [q] quit")

    try:
        with OpenCVCamera(camera_config) as camera:
            cached_overlay: tuple[np.ndarray, np.ndarray] | None = None
            cached_edges: tuple[np.ndarray, np.ndarray] | None = None
            cached_frame_shape: tuple[int, ...] | None = None
            cached_fit_mode: str | None = None
            cached_mirror_state: bool | None = None

            while True:
                try:
                    live_frame = camera.async_read(timeout_ms=1000)
                except TimeoutError:
                    continue

                reference_source = reference_image
                if viewer_state.mirror:
                    live_frame = cv2.flip(live_frame, 1)
                    reference_source = cv2.flip(reference_image, 1)

                if (
                    cached_overlay is None
                    or cached_edges is None
                    or cached_frame_shape != live_frame.shape
                    or cached_fit_mode != viewer_state.fit_mode
                    or cached_mirror_state != viewer_state.mirror
                ):
                    cached_overlay = fit_reference_to_frame(reference_source, live_frame.shape, viewer_state.fit_mode)
                    reference_edges = compute_reference_edges(
                        reference_source,
                        threshold1=args.edge_threshold1,
                        threshold2=args.edge_threshold2,
                    )
                    cached_edges = fit_reference_to_frame(
                        reference_edges,
                        live_frame.shape,
                        viewer_state.fit_mode,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    cached_frame_shape = live_frame.shape
                    cached_fit_mode = viewer_state.fit_mode
                    cached_mirror_state = viewer_state.mirror

                fitted_reference, valid_mask = cached_overlay
                fitted_edges, edge_mask = cached_edges

                opacity_percent = cv2.getTrackbarPos(ALPHA_TRACKBAR, WINDOW_NAME)
                display_frame = blend_reference_overlay(
                    live_frame=live_frame,
                    reference_frame=fitted_reference,
                    valid_mask=valid_mask,
                    alpha=opacity_percent / 100.0,
                )

                if viewer_state.show_edges:
                    display_frame = overlay_reference_edges(
                        live_frame=display_frame,
                        reference_edges=fitted_edges,
                        valid_mask=edge_mask,
                    )

                display_frame = draw_status_panel(
                    display_frame,
                    opacity_percent=opacity_percent,
                    viewer_state=viewer_state,
                )
                cv2.imshow(WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("e"):
                    viewer_state.show_edges = not viewer_state.show_edges
                elif key == ord("f"):
                    viewer_state.fit_mode = "contain" if viewer_state.fit_mode == "stretch" else "stretch"
                elif key == ord("m"):
                    viewer_state.mirror = not viewer_state.mirror
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
