"""Step 3: turn all finished runs into readable reports.

    uv run scripts/make_report.py

Reads   outputs/<variant>/<model>/fold<k>/{results.json, scores.npz} and data/manifests/*.csv
Writes  reports/README.md, reports/data.md, reports/task1_session_feasibility.md,
        reports/task2_strong_system.md, reports/figures/*.png
and logs the reports as artifacts of an MLflow run named "report".

Privacy: only aggregated numbers are written. No names, no images of our participants.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import hydra
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"

# Fixed order and colour per system (colour follows the system, never its rank).
VARIANT_ORDER = ["zero_shot", "external_only", "hagrid_external", "ours_sessions", "ours_sessions_external", "ours_only",
                 "ours_external", "ours_external_meta"]
VARIANT_LABEL = {"zero_shot": "Zero-shot (no training)", "external_only": "Trained on 11k + HD",
                 "ours_only": "Trained on ours", "ours_external": "Trained on ours + 11k + HD",
                 "ours_sessions": "Ours, session labels only",
                 "ours_sessions_external": "Ours session labels + 11k + HD",
                 "ours_external_meta": "Ours + 11k + HD + metadata heads",
                 "hagrid_external": "Trained on HaGRID + 11k + HD"}
MODEL_LABEL = {"dinov3_vitb": "DINOv3 ViT-B", "dinov3_vitb_masked": "DINOv3 ViT-B, background masked",
               "dinov3_vitl": "DINOv3 ViT-L", "clip_vitb": "CLIP ViT-B", "siglip2_vitb": "SigLIP 2 ViT-B",
               "dinov3_convnext_b": "DINOv3 ConvNeXt-B", "resnet50_in1k": "ResNet-50 (IN-1k)",
               "efficientnet_b3_in1k": "EfficientNet-B3 (IN-1k)", "convnext_b_in22k": "ConvNeXt-B (IN-22k)",
               "clip_vitl_laion": "CLIP ViT-L/14 (LAION-2B)", "clip_vith_laion": "CLIP ViT-H/14 (LAION-2B)",
               "convnext_b_clip_laion": "CLIP ConvNeXt-B (LAION-2B)",
               "convnext_xxl_clip_laion": "CLIP ConvNeXt-XXL (LAION-2B)",
               "dinov3_convnext_b_parts_mean": "DINOv3 ConvNeXt-B, parts + mean",
               "dinov3_convnext_b_parts_transformer": "DINOv3 ConvNeXt-B, parts + transformer",
               "resnet50_in1k_parts_mean": "ResNet-50, parts + mean",
               "resnet50_in1k_parts_transformer": "ResNet-50, parts + transformer",
               "dinov3_vitb_hagrid": "DINOv3 ViT-B, HaGRID pre-trained"}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Systems of the main Task 1 report (the core question: which training data?)
MAIN_SYSTEMS = [("zero_shot", "dinov3_vitb"), ("external_only", "dinov3_vitb"), ("ours_only", "dinov3_vitb"),
                ("ours_external", "dinov3_vitb")]

# One comparison report per idea / GitHub issue. The first system is the reference for paired deltas.
IDEAS = {
    "backbones": dict(issue=2, title="Backbones: CLIP and SigLIP vs DINOv3",
                      question="How much does the pre-trained backbone matter? Are image-text models (CLIP, "
                               "SigLIP 2) a good starting point for hand biometrics?",
                      systems=[("ours_external", "dinov3_vitb"), ("zero_shot", "dinov3_vitb"), ("zero_shot", "clip_vitb"),
                               ("zero_shot", "siglip2_vitb"), ("ours_external", "clip_vitb"),
                               ("ours_external", "siglip2_vitb"), ("ours_external", "dinov3_convnext_b")]),
    "session_only": dict(issue=3, title="Session-only supervision",
                         question="Can we learn to recognise people if we never label which sessions belong to "
                                  "the same person (every session = its own class)?",
                         systems=[("ours_sessions_external", "dinov3_vitb"), ("external_only", "dinov3_vitb"),
                                  ("ours_sessions", "dinov3_vitb"), ("ours_only", "dinov3_vitb"),
                                  ("ours_external", "dinov3_vitb")]),
    "masking": dict(issue=4, title="Background masking",
                    question="Is the gap between same-session and cross-session performance caused by the "
                             "background?",
                    systems=[("ours_external", "dinov3_vitb"), ("ours_external", "dinov3_vitb_masked")]),
    "metadata": dict(issue=5, title="Metadata as auxiliary targets",
                     question="Does predicting 11k metadata (gender, skin colour, accessories, nail polish, age) "
                              "from the embedding help identification?",
                     systems=[("ours_external", "dinov3_vitb"), ("ours_external_meta", "dinov3_vitb")]),
    "classic_cnns": dict(issue=9, title="Classic ImageNet CNNs",
                         question="How do supervised ImageNet CNNs (ResNet, EfficientNet, ConvNeXt) compare with "
                                  "modern self-supervised and image-text backbones?",
                         systems=[("ours_external", "dinov3_vitb"), ("zero_shot", "resnet50_in1k"),
                                  ("zero_shot", "convnext_b_in22k"), ("ours_external", "resnet50_in1k"),
                                  ("ours_external", "efficientnet_b3_in1k"), ("ours_external", "convnext_b_in22k"),
                                  ("ours_external", "dinov3_convnext_b")]),
    "laion": dict(issue=10, title="Extremely large pre-training (LAION-2B)",
                  question="Do OpenCLIP models trained on 2 billion image-text pairs give better hand features?",
                  systems=[("ours_external", "dinov3_vitb"), ("zero_shot", "clip_vitb"),
                           ("zero_shot", "convnext_b_clip_laion"), ("zero_shot", "clip_vitl_laion"),
                           ("zero_shot", "clip_vith_laion"), ("zero_shot", "convnext_xxl_clip_laion"),
                           ("ours_external", "convnext_b_clip_laion"), ("ours_external", "clip_vitl_laion")]),
    "parts": dict(issue=11, title="Local part features fused by averaging or a transformer",
                  question="Does keeping WHERE a feature was seen (a grid of local CNN features) and fusing the "
                           "parts beat global pooling?",
                  systems=[("ours_external", "dinov3_convnext_b"), ("ours_external", "dinov3_convnext_b_parts_mean"),
                           ("ours_external", "dinov3_convnext_b_parts_transformer"), ("ours_external", "resnet50_in1k"),
                           ("ours_external", "resnet50_in1k_parts_mean"),
                           ("ours_external", "resnet50_in1k_parts_transformer")]),
    "hagrid": dict(issue=8, title="HaGRID pre-training (12.7k extra identities)",
                   question="Do 12.7k extra identities from a gesture dataset improve cross-session recognition?",
                   systems=[("ours_external", "dinov3_vitb"), ("external_only", "dinov3_vitb"),
                            ("hagrid_external", "dinov3_vitb"), ("ours_external", "dinov3_vitb_hagrid")]),
    "vitl": dict(issue=6, title="Strongest system: DINOv3 ViT-L",
                 question="Does a 3.5x larger backbone with a longer schedule improve cross-session recognition?",
                 systems=[("ours_external", "dinov3_vitb"), ("ours_external", "dinov3_vitl")]),
}
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False,
})


# ---------------------------------------------------------------------------------------------
# Collect results
# ---------------------------------------------------------------------------------------------
def collect(outputs: Path) -> dict[tuple[str, str], dict]:
    """{(variant, model): {"ours": {fold: results}, "external": {name: [results per run]}, "scores": {...}}}"""
    systems: dict = defaultdict(lambda: {"ours": {}, "external": defaultdict(list), "scores": {}})
    for res_file in sorted(outputs.glob("*/*/fold*/results.json")):
        variant, model = res_file.parts[-4], res_file.parts[-3]
        sys = systems[(variant, model)]
        results = json.loads(res_file.read_text())
        for name, r in results.items():
            if name.startswith("ours/"):
                sys["ours"][name.split("/")[1]] = r
                sys["scores"][name.split("/")[1]] = res_file.with_name("scores.npz")
            else:
                sys["external"][name].append(r)
    return dict(sorted(systems.items(), key=lambda kv: (VARIANT_ORDER.index(kv[0][0]) if kv[0][0] in VARIANT_ORDER else 99, kv[0][1])))


def label(system: tuple[str, str]) -> str:
    variant, model = system
    return f"{VARIANT_LABEL.get(variant, variant)} · {MODEL_LABEL.get(model, model)}"


SHORT_VARIANT = {"zero_shot": "zero-shot", "external_only": "11k+HD", "ours_only": "ours",
                 "ours_external": "ours+11k+HD", "ours_sessions": "ours\nsessions",
                 "ours_sessions_external": "ours sessions\n+11k+HD", "ours_external_meta": "ours+11k+HD\n+meta",
                 "hagrid_external": "HaGRID\n+11k+HD"}
SHORT_MODEL = {"dinov3_vitb": "", "dinov3_vitb_masked": "\nmasked", "dinov3_vitl": "\nViT-L",
               "clip_vitb": "\nCLIP", "siglip2_vitb": "\nSigLIP 2", "dinov3_convnext_b": "\nConvNeXt"}


def short_label(system: tuple[str, str]) -> str:
    variant, model = system
    return SHORT_VARIANT.get(variant, variant) + SHORT_MODEL.get(model, f"\n{model}")


def colour(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def fold_values(sys: dict, protocol: str, metric: str) -> np.ndarray:
    return np.array([r[protocol][metric] for r in sys["ours"].values() if r.get(protocol)
                     and r[protocol].get(metric) is not None])


def mean_std(values: np.ndarray, pct: bool = True) -> str:
    if len(values) == 0:
        return "–"
    k = 100 if pct else 1
    return f"{values.mean() * k:.1f} ± {values.std() * k:.1f}" if len(values) > 1 else f"{values.mean() * k:.1f}"


def pooled_scores(sys: dict, protocol: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s, g, w = [], [], []
    for fold, npz_path in sys["scores"].items():
        z = np.load(npz_path)
        key = f"ours/{fold}|{protocol}"
        if f"{key}|score" not in z:
            continue
        s.append(z[f"{key}|score"]); g.append(z[f"{key}|genuine"]); w.append(z[f"{key}|weight"])
    return np.concatenate(s), np.concatenate(g), np.concatenate(w)


def restrict(systems: dict, wanted: list[tuple[str, str]]) -> tuple[dict, list[str]]:
    """Keep the wanted systems that exist, and only the folds that ALL of them were evaluated on."""
    present = {k: systems[k] for k in wanted if k in systems and systems[k]["ours"]}
    if not present:
        return {}, []
    folds = sorted(set.intersection(*(set(v["ours"]) for v in present.values())), key=lambda f: int(f.split("_")[1]))
    out = {k: {"ours": {f: v["ours"][f] for f in folds}, "scores": {f: v["scores"][f] for f in folds},
               "external": v["external"]} for k, v in present.items()}
    return out, folds


def paired_delta(sys: dict, ref: dict, protocol: str = "image_cross_session") -> str:
    """Mean ± std over folds of (system EER - reference EER), in percentage points."""
    d = fold_values(sys, protocol, "eer") - fold_values(ref, protocol, "eer")
    return "reference" if np.allclose(d, 0) else f"{d.mean() * 100:+.1f} ± {d.std() * 100:.1f}"


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
def fig_det(systems: dict, protocol: str, title: str, path: Path) -> None:
    """DET curve: false non-match rate vs. false match rate, both on a normal-deviate (probit)
    scale - the standard biometric plot. Lower-left is better; the dashed diagonal is FMR = FNMR,
    so a curve crosses it at its EER."""
    from scipy.stats import norm
    from sklearn.metrics import det_curve

    ticks = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6]
    tick_labels = ["0.1", "0.5", "1", "2", "5", "10", "20", "40", "60"]
    lo, hi = norm.ppf(0.0008), norm.ppf(0.7)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--")
    ax.annotate("EER line", (norm.ppf(0.3), norm.ppf(0.3)), fontsize=7, color=MUTED, rotation=45,
                ha="center", va="bottom")
    for i, (system, sys) in enumerate(systems.items()):
        if not sys["scores"]:
            continue
        s, g, w = pooled_scores(sys, protocol)
        fmr, fnmr, _ = det_curve(g, s, sample_weight=w)
        ax.plot(norm.ppf(np.clip(fmr, 1e-5, 1 - 1e-5)), norm.ppf(np.clip(fnmr, 1e-5, 1 - 1e-5)),
                color=colour(i), lw=2, label=label(system))
    ax.set_xticks(norm.ppf(ticks), tick_labels)
    ax.set_yticks(norm.ppf(ticks), tick_labels)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("False match rate (%)  - impostors accepted")
    ax.set_ylabel("False non-match rate (%)  - genuine rejected")
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7.5)
    fig.savefig(path)
    plt.close(fig)


def fig_eer_dots(systems: dict, path: Path) -> None:
    """Per-fold EER of each system (dots) with the mean (bar), for the three protocols."""
    protocols = [("image_same_session", "Image vs image,\nsame session"),
                 ("image_cross_session", "Image vs image,\ndifferent session"),
                 ("session_template", "Session vs session\n(averaged embeddings)")]
    fig, axes = plt.subplots(1, 3, figsize=(max(9, 1.6 * len(systems) * 3 / 2), 3.4), sharey=True)
    for ax, (protocol, title) in zip(axes, protocols):
        for i, (system, sys) in enumerate(systems.items()):
            v = fold_values(sys, protocol, "eer") * 100
            if len(v) == 0:
                continue
            ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=14, color=colour(i), alpha=0.6,
                       edgecolor="white", linewidth=0.5)
            ax.hlines(v.mean(), i - 0.3, i + 0.3, color=INK, lw=2)
            ax.annotate(f"{v.mean():.1f}", (i + 0.32, v.mean()), fontsize=7.5, color=INK, va="center")
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels([short_label(sy) for sy in systems], fontsize=7.5)
        ax.set_xlabel("trained on")
    axes[0].set_ylabel("EER per held-out fold (%)")
    fig.savefig(path)
    plt.close(fig)


def fig_sessions_hist(ours: pd.DataFrame, path: Path) -> None:
    per_hand = ours.groupby("identity")["session"].nunique()
    bins = [1, 2, 3, 4, 6, 11, 21, 1000]
    names = ["1", "2", "3", "4–5", "6–10", "11–20", ">20"]
    counts = pd.cut(per_hand, bins=bins, right=False, labels=names).value_counts().reindex(names)
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.bar(names, counts.values, color=PALETTE[0], width=0.7)
    for x, c in zip(names, counts.values):
        ax.annotate(str(c), (x, c), ha="center", va="bottom", fontsize=8, color=INK, xytext=(0, 2),
                    textcoords="offset points")
    ax.set_xlabel("Sessions recorded for a hand")
    ax.set_ylabel("Number of hands")
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------------------------
def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def data_report(m: dict[str, pd.DataFrame]) -> str:
    rows = []
    for ds, df in m.items():
        sess = df.groupby("identity")["session"].nunique()
        rows.append({
            "Dataset": ds, "Images": len(df), "People": df.subject.nunique(), "Hands (identities)": df.identity.nunique(),
            "Hands with ≥2 sessions": int((sess > 1).sum()) if ds == "ours" else "– (1 session)",
            "Hand detected (MediaPipe)": f"{(df['align'] != 'fallback').mean() * 100:.1f} %",
            "Split": "10 people-disjoint folds" if ds == "ours" else
                     f"train {df[df.split == 'train'].subject.nunique()} / test {df[df.split == 'test'].subject.nunique()} people",
        })
    ours = m["ours"]
    sess_per_person = ours.groupby("subject")["session"].nunique()
    per_fold = ours.groupby("split").agg(people=("subject", "nunique"), hands=("identity", "nunique"),
                                         images=("path", "size"),
                                         sessions=("session", lambda s: ours.loc[s.index].groupby(["identity", "session"]).ngroups))
    per_fold = per_fold.reindex(sorted(per_fold.index, key=lambda f: int(f.split("_")[1]))).reset_index()
    per_fold.columns = ["Fold", "People", "Hands", "Images", "Sessions"]
    imgs_per_hand = ours.identity.value_counts()
    return f"""# Data

All three datasets are converted to the same canonical form before training:
MediaPipe finds 21 hand landmarks, the image is rotated so that the fingers point up,
cropped to a square around the hand and **left hands are mirrored** to look like right
hands. Left and right hand of one person are still treated as *different* identities.

No example images are included in this repository to protect participants' privacy.

## Overview

{md_table(pd.DataFrame(rows))}

* **ours** – our dataset (`kfolds_people`), dorsal side, phone photos, varied backgrounds.
  Participants are anonymised (`ours_000` …); the name map is kept outside the repository.
* **11k** – 11k Hands (Afifi, 2019), only the *dorsal* images, white background, 1 session.
* **hd** – PolyU Contactless Hand Dorsal (Kumar & Xu, 2016), right hands, 1 session.
  External datasets are split by person; 20 % of people are a fixed test set never used for training.

## Our dataset in detail

* **Duplicates.** 67 images were byte-identical copies of images in *another session of the
  same hand* (48 originals). They were removed; otherwise "cross-session" comparisons
  would include identical images and look better than reality.
* **Sessions are very unevenly distributed.** Median {sess_per_person.median():.0f} session(s) per person,
  maximum {sess_per_person.max()}. The most recorded hand has {imgs_per_hand.max()} images, the median hand {imgs_per_hand.median():.0f}.
  Evaluation therefore weights every hand equally (see Task 1 report).

![Sessions per hand](figures/sessions_per_hand.png)

### Folds

{md_table(per_fold)}
"""


def task1_report(systems: dict) -> str:
    rows = []
    for i, (system, sys) in enumerate(systems.items()):
        if not sys["ours"]:
            continue
        rows.append({
            "#": i + 1, "System": label(system), "Folds": len(sys["ours"]),
            "EER same session": mean_std(fold_values(sys, "image_same_session", "eer")),
            "**EER cross-session**": mean_std(fold_values(sys, "image_cross_session", "eer")),
            "TAR@FAR=1 % cross-session": mean_std(fold_values(sys, "image_cross_session", "tar@far=0.01")),
            "**EER session template**": mean_std(fold_values(sys, "session_template", "eer")),
            "TAR@FAR=1 % session template": mean_std(fold_values(sys, "session_template", "tar@far=0.01")),
            "Rank-1 (session)": mean_std(fold_values(sys, "rank1_session", "rank1")),
        })
    table = md_table(pd.DataFrame(rows)) if rows else "_no runs yet_"

    fold_rows = []
    for system, sys in systems.items():
        for fold, r in sorted(sys["ours"].items(), key=lambda kv: int(kv[0].split("_")[1])):
            fold_rows.append({"System": label(system), "Fold": fold,
                              "EER cross-session": f"{r['image_cross_session']['eer'] * 100:.1f}" if r.get("image_cross_session") else "–",
                              "EER session template": f"{r['session_template']['eer'] * 100:.1f}" if r.get("session_template") else "–",
                              "Genuine cross-session pairs": r["image_cross_session"].get("n_genuine", 0) if r.get("image_cross_session") else 0,
                              "Rank-1": f"{r['rank1_session']['rank1'] * 100:.1f}" if r["rank1_session"]["rank1"] is not None else "–"})
    return f"""# Task 1 – Can our dataset be used to collect sessions for person identification?

**Question.** If a person records their hand in one session, can they be recognised
from a photo taken in another session? We measure this on people the model has
*never seen*: for each of the 10 people-disjoint folds of our dataset, the model is
trained without that fold and evaluated on it (10-fold cross-validation over people).

## What is compared

| Protocol | Genuine pair | Impostor pair |
|---|---|---|
| Image vs image, same session | two photos of the same hand from the same session | same side, different person |
| **Image vs image, cross-session** | two photos of the same hand from *different* sessions | same side, different person |
| **Session template** | average embedding of one session vs. another session of the same hand | same side, different person |
| Rank-1 (session) | gallery = first session of every hand in the fold (all hands, single-session ones act as distractors), probes = later sessions | – |

Every hand counts equally (pair weights), so the person with 79 sessions does not dominate.
Hands with only one session contribute impostor pairs only. Numbers are mean ± std over
the held-out folds; **EER** = equal error rate (lower is better), **TAR@FAR=1 %** =
share of genuine pairs accepted when 1 % of impostors are accepted (higher is better).
Rank-1 is measured within one fold (≈ 8–9 people, ≈ 15 hands), so it is an easy closed-set task.

## Results

{table}

![EER per fold](figures/task1_eer_per_fold.png)

DET curves (detection error trade-off: both error rates on a probit scale, lower-left is better,
a curve crosses the dashed diagonal at its EER). They pool the scores of all held-out folds,
each fold scored by the model that did not see it:

![DET cross-session](figures/task1_det_cross_session.png)
![DET same session](figures/task1_det_same_session.png)
![DET session template](figures/task1_det_session_template.png)

{{conclusion}}

<details><summary>Per-fold numbers</summary>

{md_table(pd.DataFrame(fold_rows)) if fold_rows else ""}

</details>
"""


def task2_report(systems: dict) -> str:
    rows = []
    for i, (system, sys) in enumerate(systems.items()):
        row = {"#": i + 1, "System": label(system), "Folds (ours)": len(sys["ours"])}
        for name, short in [("11k/test", "11k"), ("hd/test", "HD")]:
            runs = sys["external"].get(name, [])
            row[f"{short} EER"] = mean_std(np.array([r["image_all"]["eer"] for r in runs]))
            row[f"{short} rank-1"] = mean_std(np.array([r["rank1_image"]["rank1"] for r in runs]))
        row["Ours EER cross-session"] = mean_std(fold_values(sys, "image_cross_session", "eer"))
        row["Ours EER session template"] = mean_std(fold_values(sys, "session_template", "eer"))
        rows.append(row)
    return f"""# Task 2 – A strong dorsal-hand identification system

**Goal.** The best possible recogniser of people from the back of the hand, using our
dataset together with 11k Hands and PolyU HD.

## Method

1. **Alignment** (`src/handid/align.py`): MediaPipe hand landmarks → rotate fingers up →
   square crop → mirror left hands. This puts three very differently photographed
   datasets into one common form.
2. **Backbone**: DINOv3 Vision Transformer (self-supervised on 1.7 B images, via `timm`),
   fine-tuned end-to-end with layer-wise learning-rate decay, 384 px input.
3. **Loss**: ArcFace (additive angular margin) over hand identities; a 512-d L2-normalised
   embedding; hands are compared by cosine similarity, so unseen people need no retraining.
4. **Sampling**: identities drawn ∝ 1/√(images), so heavily recorded hands do not dominate.

## Test sets

* **11k / HD**: subject-disjoint 20 % test splits (38 / {{hd_test_people}} people). EER over all pairs; rank-1 with
  the first image of each hand as gallery and the rest as probes. Both datasets have a single
  session, so these numbers are *same-session* and optimistic.
* **Ours**: the 10-fold cross-session protocol from the Task 1 report.

Mean ± std is over the runs (one model per held-out fold of our data).

## Results

{md_table(pd.DataFrame(rows))}

{{conclusion}}
"""


def external_cell(sys: dict, name: str, protocol: str, metric: str) -> str:
    runs = sys["external"].get(name, [])
    return mean_std(np.array([r[protocol][metric] for r in runs]))


def experiment_report(key: str, idea: dict, systems: dict, conclusion: str) -> str:
    subset, folds = restrict(systems, idea["systems"])
    if not subset:
        return f"# {idea['title']} (#{idea['issue']})\n\n_No results yet._\n"
    ref = next(iter(subset.values()))
    rows = []
    for system, sys in subset.items():
        rows.append({
            "System": label(system),
            "**EER cross-session**": mean_std(fold_values(sys, "image_cross_session", "eer")),
            "Δ vs reference (paired)": paired_delta(sys, ref),
            "TAR@FAR=1 % cross-session": mean_std(fold_values(sys, "image_cross_session", "tar@far=0.01")),
            "EER session template": mean_std(fold_values(sys, "session_template", "eer")),
            "EER same session": mean_std(fold_values(sys, "image_same_session", "eer")),
            "Rank-1 (session)": mean_std(fold_values(sys, "rank1_session", "rank1")),
            "11k EER": external_cell(sys, "11k/test", "image_all", "eer"),
            "HD EER": external_cell(sys, "hd/test", "image_all", "eer"),
        })
    fig_det(subset, "image_cross_session", "Image vs image, different sessions", FIGS / f"exp_{key}_det.png")
    fig_eer_dots(subset, FIGS / f"exp_{key}_eer_per_fold.png")
    return f"""# {idea['title']}

GitHub issue: #{idea['issue']}

**Question.** {idea['question']}

**Protocol.** Our dataset, people-disjoint folds {', '.join(f.split('_')[1] for f in folds)}
(only folds that every system below was evaluated on, so the comparison is paired).
EER / TAR in %, mean ± std over folds; *Δ vs reference* is the per-fold difference of the
cross-session EER to the first row (negative = better). 11k / HD: subject-disjoint test splits
(single session, so optimistic).

{md_table(pd.DataFrame(rows))}

![DET](../figures/exp_{key}_det.png)
![EER per fold](../figures/exp_{key}_eer_per_fold.png)

{conclusion}
"""


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    manifests = {ds: pd.read_csv(Path(cfg.paths.manifest_dir) / f"{ds}.csv", dtype=str) for ds in ("ours", "11k", "hd")}
    all_systems = collect(ROOT / "outputs")
    systems, _ = restrict(all_systems, MAIN_SYSTEMS)
    conclusions = json.loads((REPORTS / "conclusions.json").read_text()) if (REPORTS / "conclusions.json").exists() else {}

    fig_sessions_hist(manifests["ours"], FIGS / "sessions_per_hand.png")
    fig_eer_dots(systems, FIGS / "task1_eer_per_fold.png")
    fig_det(systems, "image_same_session", "Image vs image, same session", FIGS / "task1_det_same_session.png")
    fig_det(systems, "image_cross_session", "Image vs image, different sessions", FIGS / "task1_det_cross_session.png")
    fig_det(systems, "session_template", "Session vs session (averaged embeddings)", FIGS / "task1_det_session_template.png")

    hd_test = manifests["hd"].query("split == 'test'").subject.nunique()
    (REPORTS / "data.md").write_text(data_report(manifests))
    (REPORTS / "task1_session_feasibility.md").write_text(
        task1_report(systems).replace("{conclusion}", conclusions.get("task1", "")))
    (REPORTS / "experiments").mkdir(exist_ok=True)
    for key, idea in IDEAS.items():
        (REPORTS / "experiments" / f"{key}.md").write_text(
            experiment_report(key, idea, all_systems, conclusions.get(key, "")))
    (REPORTS / "task2_strong_system.md").write_text(
        task2_report(all_systems).replace("{hd_test_people}", str(hd_test)).replace("{conclusion}", conclusions.get("task2", "")))

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)
    with mlflow.start_run(run_name="report", tags={"kind": "report"}):
        for system, sys in all_systems.items():
            for protocol in ("image_cross_session", "session_template"):
                v = fold_values(sys, protocol, "eer")
                if len(v):
                    mlflow.log_metric(f"{system[0]}.{system[1]}.{protocol}.eer_mean", float(v.mean()))
        mlflow.log_artifacts(str(REPORTS), artifact_path="reports")
    print("reports written to", REPORTS)


if __name__ == "__main__":
    main()
