"""Hand alignment: turn any photo of a hand into a canonical, square crop.

Canonical form (identical for all datasets, which removes most of the domain gap):
    * fingers point up (wrist -> middle-finger knuckle vector is vertical),
    * left hands are mirrored so every hand looks like a right hand,
    * square crop around the 21 MediaPipe landmarks with a margin,
    * resized to ``out_size`` and cached as JPEG.

If MediaPipe finds no hand we retry on the image rotated by 180 degrees, and as a last
resort fall back to the whole image (padded to a square, rotated by the dataset's usual
orientation). The method used is recorded in the manifest column ``align``.
"""

from __future__ import annotations

import math
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageOps

WRIST, MIDDLE_MCP = 0, 9  # MediaPipe landmark indices

# Rotation (degrees, counter-clockwise) that makes fingers point up when detection fails.
DEFAULT_ROTATION = {"ours": 0, "11k": 180, "hd": 180}


def load_image(path: str, max_side: int = 1280) -> Image.Image:
    """Load an RGB image, honour EXIF orientation, downscale cheaply (JPEG draft mode)."""
    im = Image.open(path)
    im.draft("RGB", (max_side, max_side))  # decodes big JPEGs at 1/2, 1/4, 1/8 size: much faster
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((max_side, max_side))
    return im


class HandAligner:
    def __init__(self, model_path: str, out_size: int = 448, margin: float = 1.3):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=1,
            min_hand_detection_confidence=0.3,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.out_size = out_size
        self.margin = margin

    def landmarks(self, im: Image.Image) -> np.ndarray | None:
        """21x2 array of landmark pixel coordinates, or None if no hand was found."""
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(im)))
        if not result.hand_landmarks:
            return None
        return np.array([[p.x * im.width, p.y * im.height] for p in result.hand_landmarks[0]])

    def crop_from_landmarks(self, im: Image.Image, pts: np.ndarray) -> Image.Image:
        # 1) rotate the image so that wrist -> middle knuckle points straight up
        dx, dy = pts[MIDDLE_MCP] - pts[WRIST]
        angle = math.degrees(math.atan2(dx, -dy))  # 0 when the vector already points up
        center = pts.mean(axis=0)
        rotated = im.rotate(angle, resample=Image.BILINEAR, center=tuple(center), fillcolor=(0, 0, 0))

        # 2) rotate the landmarks the same way and take a square box around them
        theta = math.radians(angle)
        rot = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
        pts_rot = (pts - center) @ rot.T + center
        (x0, y0), (x1, y1) = pts_rot.min(axis=0), pts_rot.max(axis=0)
        side = max(x1 - x0, y1 - y0) * self.margin
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
        return rotated.crop(box)  # areas outside the image become black

    def fallback_crop(self, im: Image.Image, dataset: str) -> Image.Image:
        im = im.rotate(DEFAULT_ROTATION[dataset], expand=True)
        return ImageOps.pad(im, (max(im.size),) * 2, color=(0, 0, 0))

    def __call__(self, path: str, dataset: str, side: str) -> tuple[Image.Image, str]:
        """Return (aligned crop, method) where method is 'mediapipe', 'mediapipe_rot180' or 'fallback'."""
        original = load_image(path)
        for method, im in (("mediapipe", original), ("mediapipe_rot180", original.rotate(180))):
            pts = self.landmarks(im)
            if pts is not None:
                crop = self.crop_from_landmarks(im, pts)
                break
        else:  # no hand found in either orientation
            method, crop = "fallback", self.fallback_crop(original, dataset)
        if side == "left":
            crop = ImageOps.mirror(crop)
        return crop.resize((self.out_size, self.out_size), Image.BICUBIC), method


# --- multiprocessing helpers (one detector per worker process) ---------------------------------
_worker: HandAligner | None = None


def init_worker(model_path: str, out_size: int, margin: float) -> None:
    global _worker
    _worker = HandAligner(model_path, out_size, margin)


def align_one(job: tuple[str, str, str, str]) -> str:
    src, dst, dataset, side = job
    try:
        crop, method = _worker(src, dataset, side)
    except Exception as e:  # unreadable file etc. -> reported, row gets dropped later
        return f"error:{type(e).__name__}"
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    crop.save(dst, quality=95)
    return method
