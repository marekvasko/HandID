# HandID – recognising people by the back of their hand

Two questions, one code base:

1. **Can our own dataset be used to collect sessions for person identification?**
   → [reports/task1_session_feasibility.md](reports/task1_session_feasibility.md)
2. **How good can a dorsal-hand recogniser get when we also use the public 11k Hands and
   PolyU HD datasets?** → [reports/task2_strong_system.md](reports/task2_strong_system.md)

Data description and quality checks: [reports/data.md](reports/data.md).
Every run is also logged to MLflow (experiment `handid-dorsal`).

## How it works

```
photo ──► MediaPipe landmarks ──► rotate fingers-up, square crop, mirror left hands
      ──► DINOv3 ViT (fine-tuned, ArcFace loss) ──► 512-d embedding
      ──► cosine similarity between two hands  ──► same person?
```

* Evaluation is always on **people the model has never seen** (10 people-disjoint folds of our
  data; subject-disjoint test splits of 11k and HD).
* The key number for our dataset is **cross-session** performance: a photo from one day
  compared with a photo from another day.

## Quick start

```bash
uv sync                                    # Python 3.12, PyTorch (CUDA 12.8), Lightning, Hydra, MLflow
# edit configs/paths/default.yaml if your data lives elsewhere
uv run scripts/prepare_data.py             # unpack, anonymise, deduplicate, align (once, ~15 min)
uv run scripts/train.py fold=0             # train + evaluate one model
bash scripts/run_all.sh                    # every experiment in the reports
uv run scripts/make_report.py              # regenerate reports/
uv run pytest                              # sanity tests of the metrics
```

Configuration is plain Hydra: `configs/config.yaml` plus the groups `model/`
(backbone) and `variant/` (training data). Any value can be overridden on the command
line, e.g. `uv run scripts/train.py variant=ours_only model=dinov3_vitl train.epochs=20`.

## Data

| Dataset | What | Where it comes from |
|---|---|---|
| ours | 87 people, both hands, many sessions, phone photos | private, not distributed |
| 11k Hands | 190 people, dorsal + palmar (we use dorsal) | Afifi, *11K Hands*, MTAP 2019 |
| PolyU HD | 714 people, right-hand dorsal | Kumar & Xu, IEEE TIFS 2016 |
| HaGRID v2 (optional) | ~12k users, gestures `stop_inverted`, `peace_inverted`, `two_up_inverted` (back of the hand) | Kapitanov et al., HaGRID, WACV 2024 |

### HaGRID pre-training (issue #8)

HaGRID ships landmarks, so crops use them directly (no MediaPipe detection); it has no
left/right label, the side is derived from geometry (`manifests.hagrid_side`). 10 % of the
users are a subject-disjoint test set (`hagrid/test`), evaluated in every run once
`data/manifests/hagrid.csv` exists.

```bash
uv run scripts/prepare_data.py datasets=[hagrid]                                   # manifest + crops
uv run scripts/train.py variant=hagrid_external model=dinov3_vitb save_model=true  # stage 1 (never sees ours)
uv run scripts/train.py -m variant=ours_external model=dinov3_vitb_hagrid fold=0,1,2,3,4  # stage 2
```

Stage 2 starts from `outputs/hagrid_external/dinov3_vitb/fold0/model.pt` (any saved model can
be used with `model.init_from=/path/to/model.pt`).

No images, manifests, embeddings or model weights are stored in this repository.
See [AGENTS.md](AGENTS.md) for the privacy rules and coding conventions.
