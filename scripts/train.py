"""Step 2: train one model and evaluate it. One call = one (variant, model, fold).

    uv run scripts/train.py variant=ours_external fold=0
    uv run scripts/train.py -m variant=ours_only fold=0,1,2,3,4,5,6,7,8,9

Writes to the Hydra output dir (outputs/<variant>/<model>/fold<k>/):
    results.json   all metrics of every test set
    scores.npz     raw comparison scores (for ROC curves in the report)
and logs the same to MLflow (experiment ``mlflow.experiment``).
"""

import json
import subprocess
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from handid import protocols
from handid.data import HandDataModule, describe, eval_loader, load_manifests, split_train_test
from handid.model import HandIDModule


@torch.no_grad()
def embed(model: HandIDModule, df: pd.DataFrame, cfg: DictConfig) -> np.ndarray:
    model.eval().cuda()
    out = []
    for images, _ in eval_loader(df, cfg.train.image_size, cfg.eval.batch_size, cfg.train.num_workers):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(images.cuda(non_blocking=True)).cpu())
    return torch.cat(out).numpy()


def flatten(results: dict, prefix: str = "") -> dict[str, float]:
    """{'ours/fold_0': {'image_cross_session': {'eer': .1}}} -> {'ours/fold_0/image_cross_session/eer': .1}"""
    flat = {}
    for k, v in results.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten(v, key))
        elif isinstance(v, (int, float)) and v is not None:
            flat[key.replace("@", "_at_").replace("=", "")] = float(v)
    return flat


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    # ---- data -------------------------------------------------------------------------------
    train_df, test_sets = split_train_test(load_manifests(cfg.paths.manifest_dir), cfg.variant.datasets, cfg.fold)
    print(describe(train_df, test_sets))

    # ---- logging ----------------------------------------------------------------------------
    run_name = f"{cfg.variant.name}-{cfg.model.name}-fold{cfg.fold}"
    logger = MLFlowLogger(experiment_name=cfg.mlflow.experiment, run_name=run_name,
                          tracking_uri=cfg.mlflow.tracking_uri,
                          tags={"variant": cfg.variant.name, "model": cfg.model.name, "fold": str(cfg.fold),
                                "git_commit": git_commit()})
    logger.log_hyperparams(flatten(OmegaConf.to_container(cfg.train), "train")
                           | {f"model/{k}": v for k, v in cfg.model.items()}
                           | {"variant/datasets": ",".join(cfg.variant.datasets), "fold": cfg.fold,
                              "n_train_images": len(train_df), "n_train_identities": train_df.identity.nunique()})

    # ---- model + training -------------------------------------------------------------------
    m = cfg.model
    model = HandIDModule(num_classes=train_df.identity.nunique(), backbone=m.backbone,
                         embedding_dim=m.embedding_dim, arcface_scale=m.arcface_scale,
                         arcface_margin=m.arcface_margin, lr_backbone=m.lr_backbone, lr_head=m.lr_head,
                         weight_decay=m.weight_decay, warmup_epochs=m.warmup_epochs, layer_decay=m.layer_decay)
    if cfg.train.epochs > 0:
        data = HandDataModule(train_df, cfg.train.image_size, cfg.train.batch_size, cfg.train.num_workers,
                              cfg.train.balance_power)
        trainer = L.Trainer(max_epochs=cfg.train.epochs, precision=cfg.train.precision, logger=logger,
                            enable_checkpointing=False, log_every_n_steps=20, default_root_dir=out_dir,
                            gradient_clip_val=1.0)
        trainer.fit(model, datamodule=data)
    if cfg.save_model:
        torch.save({"state_dict": model.net.state_dict(), "config": OmegaConf.to_container(cfg, resolve=True)},
                   out_dir / "model.pt")

    # ---- evaluation -------------------------------------------------------------------------
    results, raw_scores = {}, {}
    for name, df in test_sets.items():
        emb = embed(model, df, cfg)
        evaluate = protocols.evaluate_ours if name.startswith("ours/") else protocols.evaluate_external
        results[name], raw = evaluate(df, emb)
        for protocol, arrays in raw.items():
            for key, arr in arrays.items():
                raw_scores[f"{name}|{protocol}|{key}"] = arr
        print(name, json.dumps(results[name], indent=1))

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    np.savez_compressed(out_dir / "scores.npz", **raw_scores)
    client, run_id = logger.experiment, logger.run_id
    for k, v in flatten(results, "test").items():
        client.log_metric(run_id, k, v)
    client.log_artifact(run_id, str(out_dir / "results.json"))
    client.log_artifact(run_id, str(out_dir / ".hydra" / "config.yaml"))


if __name__ == "__main__":
    main()
