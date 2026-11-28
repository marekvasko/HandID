"""HaGRID has no left/right label: check the geometric rule on synthetic landmarks.  uv run pytest"""

import numpy as np

from handid.manifests import hagrid_side


def dorsal_right_hand() -> np.ndarray:
    """Back of a right hand, fingers up (image coordinates, y down): thumb on the left."""
    pts = np.zeros((21, 2))
    pts[0] = (0.5, 0.9)                    # wrist
    for finger, x in enumerate((0.30, 0.40, 0.50, 0.60, 0.70)):  # thumb ... little finger
        for k in range(4):
            pts[1 + 4 * finger + k] = (x, 0.6 - 0.1 * k)
    return pts


def rotate(pts: np.ndarray, degrees: float) -> np.ndarray:
    t = np.radians(degrees)
    r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (pts - 0.5) @ r.T + 0.5


def test_side_from_geometry():
    right = dorsal_right_hand()
    left = right * [-1, 1] + [1, 0]        # mirror image = a left hand
    for degrees in (0, 45, 90, 180, 270):  # the rule must not depend on the hand's rotation ...
        assert hagrid_side(rotate(right, degrees)) == "right"
        assert hagrid_side(rotate(left, degrees)) == "left"
    assert hagrid_side(right * [16 / 9, 1]) == "right"  # ... nor on the aspect ratio
