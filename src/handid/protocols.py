"""Evaluation protocols: which comparisons are made and how they are labelled.

Rules shared by all protocols
    * Comparisons are only made between hands of the same side (left vs left, right vs right).
    * genuine  = same identity (same person, same hand)
    * impostor = different person, same side
    * Weights make every identity (genuine) / identity pair (impostor) count equally, so a
      person with 79 sessions does not dominate the error rates.

Protocols
    image_same_session   genuine pairs from the same capture session (easy, upper bound)
    image_cross_session  genuine pairs from different sessions  <- headline for our dataset
    session_template     mean embedding of a session vs. another session of the same hand
                         (how an enrol-once / verify-later system would be used)
    rank1_session        closed-set identification: gallery = first session of each hand,
                         probes = all later sessions
    image_all            all genuine pairs (external datasets have one session only)
    rank1_image          gallery = first image of each hand, probes = the remaining images
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from handid import metrics


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _pairs(meta: pd.DataFrame, emb: np.ndarray) -> dict[str, np.ndarray]:
    """All unordered pairs i<j of the same side, with their labels and balancing weights."""
    i, j = np.triu_indices(len(meta), k=1)
    side = meta["side"].to_numpy()
    keep = side[i] == side[j]
    i, j = i[keep], j[keep]

    ident = meta["identity"].to_numpy()
    subj = meta["subject"].to_numpy()
    sess = meta["session"].to_numpy()
    same_id = ident[i] == ident[j]
    impostor = subj[i] != subj[j]
    same_sess = same_id & (sess[i] == sess[j])

    # balancing weights: 1 / (number of pairs in the same group)
    counts = meta["identity"].value_counts()
    n = counts.reindex(ident).to_numpy().astype(float)
    w_impostor = 1.0 / (n[i] * n[j])
    w_genuine = np.zeros(len(i))
    for mask in (same_id & same_sess, same_id & ~same_sess):
        groups = pd.Series(ident[i][mask])
        w_genuine[mask] = 1.0 / groups.map(groups.value_counts()).to_numpy()

    # one N x N matrix product instead of gathering emb[i], emb[j]: with ~10k HaGRID test images the
    # gathered pair embeddings alone would need ~40 GB
    gram = emb @ emb.T
    return dict(score=gram[i, j], same_id=same_id, same_sess=same_sess,
                impostor=impostor, weight=np.where(same_id, w_genuine, w_impostor))


def _verification(p: dict[str, np.ndarray], genuine_mask: np.ndarray) -> tuple[dict, dict]:
    use = genuine_mask | p["impostor"]
    scores, genuine, weights = p["score"][use], genuine_mask[use], p["weight"][use]
    if genuine.sum() == 0:
        return {}, {}
    m = metrics.verification_metrics(scores, genuine, weights)
    return m, dict(score=scores, genuine=genuine, weight=weights)


def session_templates(meta: pd.DataFrame, emb: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """One L2-normalised mean embedding per (identity, session)."""
    keys = ["subject", "side", "identity", "session"]
    groups = meta.reset_index(drop=True).groupby(keys, sort=True).indices
    t_meta = pd.DataFrame(list(groups.keys()), columns=keys)
    t_emb = _l2(np.stack([emb[idx].mean(axis=0) for idx in groups.values()]))
    return t_meta, t_emb


def _session_order(s: str) -> tuple:
    return (0, int(s)) if s.isdigit() else (1, s)


def _rank1(meta: pd.DataFrame, emb: np.ndarray, first_key) -> float | None:
    """Gallery = the first item (by ``first_key``) of each identity, probes = the rest. Per side."""
    hits, total = 0, 0
    meta = meta.reset_index(drop=True)
    for _, side_df in meta.groupby("side"):
        order = side_df.assign(_k=side_df.apply(first_key, axis=1)).sort_values("_k")
        gallery_idx = order.groupby("identity").head(1).index.to_numpy()
        probe_idx = np.setdiff1d(side_df.index.to_numpy(), gallery_idx)
        if len(probe_idx) == 0:
            continue
        acc = metrics.rank1(emb[gallery_idx], meta.identity.to_numpy()[gallery_idx],
                            emb[probe_idx], meta.identity.to_numpy()[probe_idx])
        hits += acc * len(probe_idx)
        total += len(probe_idx)
    return hits / total if total else None


def evaluate_ours(meta: pd.DataFrame, emb: np.ndarray) -> tuple[dict, dict]:
    """Protocols for a held-out fold of our dataset. Returns (metrics, raw scores for ROC plots)."""
    emb = _l2(emb)
    results, raw = {}, {}

    p = _pairs(meta, emb)
    for name, mask in [("image_same_session", p["same_id"] & p["same_sess"]),
                       ("image_cross_session", p["same_id"] & ~p["same_sess"])]:
        results[name], raw[name] = _verification(p, mask)

    t_meta, t_emb = session_templates(meta, emb)
    tp = _pairs(t_meta, t_emb)
    results["session_template"], raw["session_template"] = _verification(tp, tp["same_id"])

    # hands with a single session are gallery-only distractors
    r1 = _rank1(t_meta, t_emb, lambda r: _session_order(r["session"]))
    results["rank1_session"] = {"rank1": r1, "n_gallery": int(t_meta.identity.nunique()),
                                "n_probe": int(len(t_meta) - t_meta.identity.nunique())}
    return results, raw


def evaluate_external(meta: pd.DataFrame, emb: np.ndarray) -> tuple[dict, dict]:
    """Protocols for the subject-disjoint test split of 11k / HD (single session)."""
    emb = _l2(emb)
    p = _pairs(meta, emb)
    results, raw = {}, {}
    results["image_all"], raw["image_all"] = _verification(p, p["same_id"])
    results["rank1_image"] = {"rank1": _rank1(meta, emb, lambda r: r["path"]),
                              "n_gallery": int(meta.identity.nunique()),
                              "n_probe": int(len(meta) - meta.identity.nunique())}
    return results, raw
