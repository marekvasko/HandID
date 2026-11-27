"""Audit the session labels of our dataset with a trained model.

    uv run scripts/audit_sessions.py +model_path=outputs/.../model.pt

For every session (all photos of one hand from one capture) we compute its average
embedding and ask:
    * does it match the *other sessions of the same hand*?           (own_sim)
    * does it match the *other hand of the same person* better?      (other_side_sim) -> left/right swapped?
    * does it match a *different person* better?                     (best_impostor_sim) -> wrong person?

Use a model that was NOT trained on our dataset (e.g. variant=external_only), otherwise it
has memorised our labels and cannot question them.

Output: data/private/session_audit.csv (contains real names, gitignored - never commit)
and a printed summary with anonymous ids only.
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from handid.data import eval_loader, load_manifests
from handid.model import HandEmbedder
from handid.protocols import session_templates


@torch.no_grad()
def embed_all(net: HandEmbedder, df: pd.DataFrame, cfg: DictConfig) -> np.ndarray:
    net.eval().cuda()
    out = []
    for images, _ in eval_loader(df, cfg.train.image_size, cfg.eval.batch_size, cfg.train.num_workers):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.nn.functional.normalize(net(images.cuda()).float(), dim=1).cpu())
    return torch.cat(out).numpy()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    checkpoint = torch.load(cfg.model_path, map_location="cpu")
    mcfg = checkpoint["config"]["model"]
    net = HandEmbedder(mcfg["backbone"], mcfg["embedding_dim"], pretrained=False)
    net.load_state_dict(checkpoint["state_dict"])

    ours = load_manifests(cfg.paths.manifest_dir)["ours"]
    t_meta, t_emb = session_templates(ours, embed_all(net, ours, cfg))
    sims = t_emb @ t_emb.T
    subj, side, ident = (t_meta[c].to_numpy() for c in ("subject", "side", "identity"))

    # threshold: similarity that only 1 % of impostor session pairs exceed
    i, j = np.triu_indices(len(t_meta), 1)
    impostor = (subj[i] != subj[j]) & (side[i] == side[j])
    threshold = np.quantile(sims[i[impostor], j[impostor]], 0.99)

    rows = []
    for k in range(len(t_meta)):
        own = (ident == ident[k]) & (np.arange(len(t_meta)) != k)
        other_side = (subj == subj[k]) & (side != side[k])
        others = (subj != subj[k]) & (side == side[k])
        best = np.argmax(np.where(others, sims[k], -np.inf))
        rows.append(dict(
            own_sim=sims[k, own].mean() if own.any() else np.nan,
            other_side_sim=sims[k, other_side].mean() if other_side.any() else np.nan,
            best_impostor_sim=sims[k, best], best_impostor=ident[best],
        ))
    audit = pd.concat([t_meta, pd.DataFrame(rows)], axis=1)
    audit["n_images"] = ours.groupby(["identity", "session"]).size().reindex(
        pd.MultiIndex.from_frame(t_meta[["identity", "session"]])).to_numpy()
    audit["flag_no_match"] = audit.own_sim < threshold
    audit["flag_side_swap?"] = audit.other_side_sim > audit.own_sim
    audit["flag_other_person?"] = audit.best_impostor_sim > audit.own_sim

    names = pd.read_csv(Path(cfg.paths.private_dir) / "ours_name_to_id.csv")
    audit = audit.merge(names, on="subject", how="left")
    out = Path(cfg.paths.private_dir) / "session_audit.csv"
    audit.sort_values("own_sim").to_csv(out, index=False)

    multi = audit.dropna(subset=["own_sim"])
    print(f"impostor 1% threshold = {threshold:.3f}")
    print(f"sessions of hands with >=2 sessions: {len(multi)}")
    for flag in ("flag_no_match", "flag_side_swap?", "flag_other_person?"):
        print(f"  {flag}: {int(multi[flag].sum())}")
    print(multi.sort_values("own_sim").head(15)[["identity", "session", "n_images", "own_sim", "other_side_sim",
                                                   "best_impostor_sim"]].round(3).to_string(index=False))
    print("full table (with names, private):", out)


if __name__ == "__main__":
    main()
