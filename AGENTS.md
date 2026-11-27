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
  queue.sh               step 2 as a GPU queue over configs/queue.txt (re-read before every run)
  audit_sessions.py      flags sessions that do not match their own hand (label errors)
  make_report.py         step 3: reports/*.md + figures, uploaded to MLflow
tests/                   pytest sanity checks of the evaluation code
reports/                 generated reports (markdown, rendered by GitHub)
```

## Workflow: idea -> issue -> branch -> queued runs -> report -> PR

Every change of behaviour or every experiment follows the same path, so each result can be
traced from a GitHub issue to the exact MLflow runs that produced it.

1. **Issue.** Open a GitHub issue per idea with the question, the hypothesis and the planned
   runs (variants, models, folds). Label it `experiment` or `performance`.
2. **Branch.** One branch per idea: `exp/<idea>` or `perf/<topic>`, created from `main`
   (or stacked on the branch it needs; the PR then targets that branch).
3. **Smoke test** with tiny settings and the dev MLflow experiment, never on the real one:
   `train.epochs=1 train.limit_train_batches=20 train.batch_size=16 mlflow.experiment=handid-dev
   hydra.run.dir=outputs_dev/<name>`. Do not edit code in the checkout the GPU queue runs from;
   use a `git worktree` and merge when tested (the queue imports the code at every run start).
4. **Queue the real runs** by appending lines to `configs/queue.txt`. One GPU job at a time:
   two trainings on one GPU run out of memory and invalidate both.
5. **Report.** `uv run scripts/make_report.py` writes `reports/experiments/<idea>.md` from
   the finished runs. Add the written conclusion to `reports/conclusions.json`.
6. **PR** with the results (rules below), `Closes #<issue>`.
7. **Merge only with explicit approval.** A PR is merged only after the repository owner
   explicitly approves merging *that* PR. Approval of one PR never carries over to another one;
   agents never merge on their own initiative.

### Pre-defined test configuration (never change it inside an experiment)

* **Our dataset:** people-disjoint folds `fold_0 … fold_9` from `kfolds_people`. A run with
  `fold=k` never sees fold k in training. Main systems use all 10 folds; ablations use folds
  0-4 and are compared *paired* with the reference system on exactly the same folds.
* **External:** fixed subject-disjoint test splits of 11k (38 people) and HD (143 people),
  `external_test_fraction=0.2`, `seed=0`. Never used for training or tuning.
* **Protocols** (`src/handid/protocols.py`): image-vs-image same-session and **cross-session**
  (headline), session templates, rank-1; same side only; every hand weighted equally.
* **Metrics:** EER and TAR@FAR=1 % / 0.1 % (mean ± std over folds), rank-1; DET curves pool
  the held-out scores of all folds.
* **Training defaults** (`configs/config.yaml`): 384 px, batch 64, 15 epochs, seed 0, bf16.
  If an experiment needs a different value, it is part of the experiment and must be stated.
* No hyper-parameter is chosen by looking at a test fold. If tuning is needed, tune on the
  training folds (e.g. hold out one training fold) and say so in the PR.

### Rules for experiment PRs (`exp/*`)

An experiment PR must contain **the full report in the PR description itself** (not only a
link to a file): paste `reports/experiments/<idea>.md` as the body, with figure paths turned
into absolute GitHub URLs so they render, e.g.
`gh pr create --body-file <(sed 's#](\.\./figures/#](https://github.com/<owner>/HandID/raw/<branch>/reports/figures/#g' reports/experiments/<idea>.md)`.
The description must contain:
1. The question and a link to the issue.
2. **Exact results on the pre-defined test configuration**: the table from
   `reports/experiments/<idea>.md` (EER cross-session with the paired Δ to the reference,
   TAR@FAR=1 %, session-template EER, same-session EER, rank-1, 11k / HD EER), with the folds used.
3. **Links to every MLflow run** behind the numbers (one per system and fold) and to the
   MLflow experiment. `make_report.py` writes them into the report's "Runs" section.
4. The git commit the runs were trained with (MLflow tag `git_commit`) and any setting that
   deviates from the defaults.
5. A conclusion that a reader can act on, including negative results ("no improvement" is a result).

### Rules for performance PRs (`perf/*`)

The measurements and their tables go into the PR description itself.
A performance PR must show a **measurable improvement of the same configuration on the same
machine and setup**, before vs after, measured back to back:
* same model, image size, batch size, precision, number of workers, data and GPU,
* the measured quantities: training throughput (it/s or images/s), time per step split into
  data loading / augmentation / forward+backward, sustained GPU utilisation
  (`nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1` during training), peak memory,
* the exact commands used, and the machine (GPU, CPU cores, other load),
* a check that results did not change beyond noise (same EER on one fold) or an explanation why
  they may (e.g. a different augmentation).

## Common commands

```bash
uv sync                                               # install
uv run scripts/prepare_data.py                        # once
uv run scripts/train.py variant=ours_external fold=3  # one run
setsid nohup bash scripts/queue.sh > outputs_dev/queue.log 2>&1 &   # run everything in configs/queue.txt
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
