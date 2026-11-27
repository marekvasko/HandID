"""Verification and identification metrics.

All functions take plain numpy arrays so they are easy to test and reuse.

Verification: a list of comparison scores (cosine similarity) with a label
``genuine`` (same identity) or impostor, and optional per-pair weights.
Weights let every identity contribute equally even if one person has 79 sessions.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def verification_metrics(scores: np.ndarray, genuine: np.ndarray, weights: np.ndarray | None = None,
                         fars: tuple[float, ...] = (1e-2, 1e-3)) -> dict[str, float]:
    """EER and TAR at fixed FAR (all as fractions in [0, 1])."""
    far, tar, _ = roc_curve(genuine, scores, sample_weight=weights)
    frr = 1 - tar
    i = np.nanargmin(np.abs(far - frr))
    out = {"eer": float((far[i] + frr[i]) / 2)}
    for f in fars:
        # best TAR achievable while keeping FAR <= f
        out[f"tar@far={f:g}"] = float(tar[far <= f].max()) if np.any(far <= f) else 0.0
    out["n_genuine"] = int(genuine.sum())
    out["n_impostor"] = int((~genuine).sum())
    return out


def roc_points(scores: np.ndarray, genuine: np.ndarray, weights: np.ndarray | None = None):
    far, tar, _ = roc_curve(genuine, scores, sample_weight=weights)
    return far, tar


def rank1(gallery_emb: np.ndarray, gallery_ids: np.ndarray, probe_emb: np.ndarray, probe_ids: np.ndarray) -> float:
    """Closed-set identification: fraction of probes whose most similar gallery item has the right id."""
    sims = probe_emb @ gallery_emb.T
    return float(np.mean(gallery_ids[sims.argmax(axis=1)] == probe_ids))
