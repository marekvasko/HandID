# AGENTS.md – guidelines for humans and coding agents

This repository identifies people from the **dorsal (back) side of the hand**.
Read this file before changing anything.

## Golden rules

1. **Privacy first.** Our dataset contains real people.
   * Never commit images, manifests, embeddings, checkpoints or anything under `data/` or `outputs/`.
   * Never write a participant's real name anywhere (code, configs, reports, MLflow run names,
     plots). Use the anonymous ids `ours_000 …` produced by `src/handid/manifests.py`.
     The name→id map lives only in `data/private/` (gitignored).
   * Reports contain aggregated numbers only – no example images of participants.
   * Never commit credentials. MLflow reads them from `~/.mlflow/credentials`.
2. **Readability over cleverness.** Plain PyTorch + Lightning + Hydra. One idea per function,
   docstrings that explain *why*, no framework-of-frameworks. A new contributor should
   understand a file in one read.
3. **Every number in a report must be reproducible** from a config + git commit.
   Training runs log their config, commit and metrics to MLflow.
4. **Evaluate on people the model has never seen.** Our dataset is always split by person
   (`fold_0 … fold_9`), external datasets by subject (fixed 20 % test split). Never tune on a
   test fold.

## Layout

```
configs/                 Hydra configs
  config.yaml            training/evaluation defaults (fold, epochs, image size, MLflow)
  prepare.yaml           data preparation
  paths/default.yaml     machine-specific paths – the only file to change on a new machine
  model/*.yaml           backbones + optimiser settings
  variant/*.yaml         which datasets are used for training
src/handid/
  manifests.py           one CSV per dataset with a common schema (anonymisation, dedup)
  align.py               MediaPipe landmarks -> upright, square, right-hand canonical crop
  data.py                Dataset, sampler, train/test split per variant + fold
  model.py               timm backbone + embedding head + ArcFace loss (LightningModule)
  metrics.py             EER, TAR@FAR, rank-1
  protocols.py           which comparisons are made (same/cross-session, session templates)
scripts/
  prepare_data.py        step 1: unpack, build manifests, cache aligned crops
  train.py               step 2: train one (variant, model, fold) and evaluate
  run_all.sh             step 2 for every experiment in the reports
  make_report.py         step 3: reports/*.md + figures, uploaded to MLflow
tests/                   pytest sanity checks of the evaluation code
reports/                 generated reports (markdown, rendered by GitHub)
```

## Common commands

```bash
uv sync                                               # install
uv run scripts/prepare_data.py                        # once
uv run scripts/train.py variant=ours_external fold=3  # one run
uv run scripts/train.py -m variant=ours_only fold=0,1,2,3,4,5,6,7,8,9
uv run scripts/make_report.py
uv run pytest
```

## Conventions

* Identity = `(subject, side)`. Left and right hand of one person are different identities.
  Comparisons are only made between hands of the same side.
* Metrics are fractions in code (`0.05`) and percentages in reports (`5.0 %`).
* Paths are absolute in configs (Hydra `job.chdir=false`); never rely on the working directory.
* Each run writes `outputs/<variant>/<model>/fold<k>/{results.json,scores.npz}`; the report
  script only reads those files, so reports can be regenerated without retraining.
* Add a test in `tests/` when you change `metrics.py` or `protocols.py`.
* Keep functions short; prefer a clear name over a comment, and a comment over a clever trick.
