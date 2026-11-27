"""Audit the session labels of our dataset.

    uv run scripts/audit_sessions.py                       # uses outputs/ours_only/dinov3_vitb
    uv run scripts/audit_sessions.py +runs=outputs/ours_external/dinov3_vitb

Every fold is judged by the model that did NOT see it during training (its embeddings are
saved by train.py as embeddings.npz). For every session (all photos of one hand from one
capture) we average its embeddings and ask:

    own_sim            similarity to the other sessions of the same hand
    other_side_sim     similarity to the other hand of the same person   -> left/right swapped?
    best_impostor_sim  similarity to the closest session of another person -> wrong person?

A session is flagged when it does not reach the fold's 1 %-FAR threshold with its own hand,
or when the other hand / another person matches it better than its own hand does.

Output: data/private/session_audit.csv (real names, gitignored - never commit) and a summary
with anonymous ids only.
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from handid.protocols import session_templates


def audit_fold(meta: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    t_meta, t_emb = session_templates(meta, emb)
    sims = t_emb @ t_emb.T
    subj, side, ident = (t_meta[c].to_numpy() for c in ("subject", "side", "identity"))
    i, j = np.triu_indices(len(t_meta), 1)
    impostor = (subj[i] != subj[j]) & (side[i] == side[j])
    threshold = np.quantile(sims[i[impostor], j[impostor]], 0.99)

    rows = []
    for k in range(len(t_meta)):
        own = (ident == ident[k]) & (np.arange(len(t_meta)) != k)
        other_side = (subj == subj[k]) & (side != side[k])
        others = (subj != subj[k]) & (side == side[k])
        best = int(np.argmax(np.where(others, sims[k], -np.inf)))
        rows.append(dict(own_sim=sims[k, own].mean() if own.any() else np.nan,
                         other_side_sim=sims[k, other_side].mean() if other_side.any() else np.nan,
                         best_impostor_sim=sims[k, best], best_impostor=ident[best], threshold=threshold))
    return pd.concat([t_meta, pd.DataFrame(rows)], axis=1)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    runs = Path(cfg.get("runs", "outputs/ours_only/dinov3_vitb"))
    ours = pd.read_csv(Path(cfg.paths.manifest_dir) / "ours.csv", dtype=str)
    parts = []
    for fold in sorted(ours.split.unique()):
        emb_file = runs / f"fold{fold.split('_')[1]}" / "embeddings.npz"
        meta = ours[ours.split == fold].reset_index(drop=True)  # same row order as in train.py
        parts.append(audit_fold(meta, np.load(emb_file)[f"ours/{fold}"]).assign(fold=fold))
    audit = pd.concat(parts, ignore_index=True)

    audit["flag_no_match"] = audit.own_sim < audit.threshold
    audit["flag_other_side_closer"] = audit.other_side_sim > audit.own_sim
    audit["flag_other_person_closer"] = audit.best_impostor_sim > audit.own_sim

    names = pd.read_csv(Path(cfg.paths.private_dir) / "ours_name_to_id.csv")
    out = Path(cfg.paths.private_dir) / "session_audit.csv"
    audit.merge(names, on="subject", how="left").sort_values("own_sim").to_csv(out, index=False)

    multi = audit.dropna(subset=["own_sim"])
    summary = {"sessions (hands with >=2 sessions)": len(multi),
               **{flag: int(multi[flag].sum()) for flag in audit.columns if flag.startswith("flag_")}}
    print(pd.Series(summary).to_string())
    print(multi.sort_values("own_sim").head(15)[["identity", "session", "own_sim", "other_side_sim",
                                                   "best_impostor_sim", "threshold"]].round(3).to_string(index=False))
    pd.Series(summary).to_json(Path("reports") / "session_audit_summary.json", indent=1)
    print("full table (with names, private):", out)


if __name__ == "__main__":
    main()
