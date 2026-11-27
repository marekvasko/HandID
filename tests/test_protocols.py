"""Sanity checks of the evaluation code on synthetic embeddings.  uv run pytest"""

import numpy as np
import pandas as pd

from handid import metrics, protocols


def toy_fold(noise: float, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """6 people x 2 hands x 3 sessions x 2 images; embedding = identity vector + noise."""
    rng = np.random.default_rng(seed)
    rows, emb = [], []
    for p in range(6):
        for side in ("left", "right"):
            centre = rng.normal(size=32)
            for s in range(3):
                for k in range(2):
                    rows.append(dict(subject=f"p{p}", side=side, identity=f"p{p}_{side}", session=str(s + 1),
                                     path=f"{p}/{side}/{s}/{k}.jpg"))
                    emb.append(centre + noise * rng.normal(size=32))
    return pd.DataFrame(rows), np.array(emb)


def test_perfect_separation():
    res, _ = protocols.evaluate_ours(*toy_fold(noise=0.01))
    for name in ("image_same_session", "image_cross_session", "session_template"):
        assert res[name]["eer"] == 0.0
        assert res[name]["tar@far=0.01"] == 1.0
    assert res["rank1_session"]["rank1"] == 1.0


def test_random_embeddings_are_near_chance():
    res, _ = protocols.evaluate_ours(*toy_fold(noise=100.0))
    assert 0.3 < res["image_cross_session"]["eer"] < 0.7


def test_pair_counts():
    meta, emb = toy_fold(noise=1.0)
    _, raw = protocols.evaluate_ours(meta, emb)
    cross = raw["image_cross_session"]
    # genuine cross-session pairs per hand: 6 images -> 15 pairs, minus 3 same-session pairs = 12
    assert cross["genuine"].sum() == 12 * 12
    # impostors: same side, different person: per side C(36,2) - 6*C(6,2)
    assert (~cross["genuine"]).sum() == 2 * (36 * 35 // 2 - 6 * 15)


def test_eer_simple():
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.75])
    genuine = np.array([True, True, True, False, False, False])
    assert abs(metrics.verification_metrics(scores, genuine)["eer"] - 1 / 3) < 1e-6
