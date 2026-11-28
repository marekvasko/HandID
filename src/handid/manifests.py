"""Build one manifest CSV per dataset.

Every manifest has the same columns, so the rest of the code never needs to know
which dataset an image came from:

    dataset     ours | 11k | hd | hagrid
    subject     anonymous subject id, e.g. "ours_007", "11k_0000123", "hd_0042", "hagrid_3fa2..."
    side        left | right        (which hand)
    identity    f"{subject}_{side}" (the class we learn; left and right hands are different biometrics)
    session     capture session id (string). 11k / HD have a single session "0";
                HaGRID uses the gesture name (each gesture was recorded separately).
    split       ours: "fold_0".."fold_9" (people-disjoint folds, given)
                external: "train" | "test" (subject-disjoint, fixed seed)
    src_path    absolute path of the original image

HaGRID additionally keeps its meta columns ``age``, ``gender``, ``race`` (for issue #5).

PRIVACY: the real names in our dataset are replaced by ``ours_XXX``. The mapping is
written to ``paths.private_dir`` (gitignored) and must never be committed or logged.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

COLUMNS = ["dataset", "subject", "side", "identity", "session", "split", "src_path"]


def _finish(df: pd.DataFrame, extra_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    df["identity"] = df["subject"] + "_" + df["side"]
    columns = COLUMNS + list(extra_columns)
    return df[columns].sort_values(["subject", "side", "session", "src_path"]).reset_index(drop=True)


def _subject_split(subjects: list[str], test_fraction: float, seed: int) -> dict[str, str]:
    """Assign each subject to train/test. Deterministic for a given seed."""
    subjects = sorted(subjects)
    random.Random(seed).shuffle(subjects)
    n_test = round(len(subjects) * test_fraction)
    return {s: ("test" if i < n_test else "train") for i, s in enumerate(subjects)}


def build_ours(root: Path, private_dir: Path, label_fixes: Path | None = None) -> pd.DataFrame:
    """Our dataset: <root>/fold_k/<person name>/<left|right>/<session>/<image>.jpg"""
    rows = []
    for path in sorted(root.rglob("*.jpg")):
        fold, name, side, session, _ = path.relative_to(root).parts
        rows.append(dict(dataset="ours", name=name, side=side, session=session, split=fold, src_path=str(path)))
    df = pd.DataFrame(rows)

    # Anonymize: sorted real names -> ours_000, ours_001, ...
    names = sorted(df["name"].unique())
    name_to_id = {n: f"ours_{i:03d}" for i, n in enumerate(names)}
    private_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(name_to_id, name="subject").rename_axis("name").to_csv(private_dir / "ours_name_to_id.csv")
    df["subject"] = df["name"].map(name_to_id)

    if label_fixes is not None and label_fixes.exists():
        df = apply_side_swaps(df, yaml.safe_load(label_fixes.read_text()).get("side_swaps", []))
    df = _drop_exact_duplicates(df)
    return _finish(df)


def apply_side_swaps(df: pd.DataFrame, swaps: list[dict]) -> pd.DataFrame:
    """Move sessions that were stored under the wrong hand to the other hand (configs/label_fixes.yaml)."""
    df = df.copy()
    for fix in swaps:
        rows = (df.subject == fix["subject"]) & (df.side == fix["side"]) & (df.session == str(fix["session"]))
        assert rows.any(), f"label fix matches no images: {fix}"
        df.loc[rows, "side"] = "right" if fix["side"] == "left" else "left"
        print(f"[ours] label fix: {fix['subject']} session {fix['session']} {fix['side']} -> other hand ({rows.sum()} images)")
    return df


def _drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Byte-identical images would make 'cross-session' pairs trivially similar.

    Keep only the first occurrence (manifest order) of every file hash.
    """
    def md5(p: str) -> str:
        return hashlib.md5(Path(p).read_bytes()).hexdigest()

    df = df.copy()
    df["md5"] = [md5(p) for p in df["src_path"]]
    n_before = len(df)
    df = df.drop_duplicates("md5", keep="first")
    print(f"[ours] dropped {n_before - len(df)} byte-identical duplicate images")
    return df


def build_11k(external_dir: Path, test_fraction: float, seed: int) -> pd.DataFrame:
    """11k Hands (Afifi 2017): dorsal images only. HandInfo.csv gives id + 'dorsal left/right'."""
    info = pd.read_csv(external_dir / "11k" / "HandInfo.csv", dtype={"id": str})
    info = info[info["aspectOfHand"].str.startswith("dorsal")]
    df = pd.DataFrame(dict(
        dataset="11k",
        subject="11k_" + info["id"],
        side=info["aspectOfHand"].str.split().str[1],
        session="0",
        src_path=[str(external_dir / "11k" / "Hands" / f) for f in info["imageName"]],
    ))
    df["split"] = df["subject"].map(_subject_split(df["subject"].unique().tolist(), test_fraction, seed))
    return _finish(df)


def build_hd(external_dir: Path, test_fraction: float, seed: int) -> pd.DataFrame:
    """PolyU Contactless Hand Dorsal (HD): right hands; subject id = filename prefix.

    Original images are spread over two folders (1-501 and 502-712) but the prefix is a
    global subject id (one subject appears in both folders), so we key on the prefix only.
    """
    root = external_dir / "hd" / "HD" / "Original Images"
    rows = []
    for path in sorted(root.glob("*/*")):
        if path.suffix.lower() != ".jpg":
            continue
        subject_number = int(path.name.split("_")[0])
        rows.append(dict(dataset="hd", subject=f"hd_{subject_number:04d}", side="right", session="0", src_path=str(path)))
    df = pd.DataFrame(rows)
    df["split"] = df["subject"].map(_subject_split(df["subject"].unique().tolist(), test_fraction, seed))
    return _finish(df)


# --- HaGRID ------------------------------------------------------------------------------------
HAGRID_GESTURES = ("stop_inverted", "peace_inverted", "two_up_inverted")  # back of the hand faces the camera


def hagrid_side(landmarks: np.ndarray) -> str:
    """Which hand a *dorsal* view shows, from the 21 landmarks alone (HaGRID has no left/right label).

    Rotate the hand so that wrist (0) -> middle knuckle (9) points up. Looking at the back of
    a RIGHT hand, the thumb side (landmarks 1 / 5) is then on the left, i.e. at a smaller x than
    the little-finger side (17); for a left hand it is the other way round.

    "Rotate, then compare x" is the sign of the 2-D cross product up x (little - index knuckle),
    which needs no rotation and does not depend on the image aspect ratio (normalised x and y
    are scaled differently, a positive scaling never flips the sign).

    Many HaGRID images are mirrored selfies, so the derived side may be the opposite of the real
    hand. That does not matter for identification: a user's camera mirrors all of their images
    the same way, so the label is swapped consistently and (user, side) stays one physical hand.
    """
    up = landmarks[9] - landmarks[0]
    across = landmarks[17] - landmarks[5]
    cross = up[0] * across[1] - up[1] * across[0]  # image coordinates: y points down
    return "right" if cross > 0 else "left"


def _hagrid_rows(annotation_file: Path, gesture: str, image_dir: Path, dropped: dict[str, int]) -> list[dict]:
    rows = []
    for uuid, a in json.loads(annotation_file.read_text()).items():
        hand = a["labels"].index(gesture)  # the other hand, if any, is labelled "no_gesture"
        landmarks = a["hand_landmarks"][hand] if hand < len(a["hand_landmarks"]) else []
        if len(landmarks) != 21:
            dropped["no landmarks"] += 1
            continue
        src = image_dir / gesture / f"{uuid}.jpg"
        if not src.exists():
            dropped["image missing"] += 1
            continue
        meta = {k: (v[hand] if len(v) == len(a["labels"]) else v[0]) if v else None for k, v in a["meta"].items()}
        rows.append(dict(dataset="hagrid", subject="hagrid_" + a["user_id"][:16],
                         side=hagrid_side(np.asarray(landmarks)), session=gesture, src_path=str(src),
                         landmarks=json.dumps(np.round(landmarks, 5).tolist()), **meta))
    return rows


def build_hagrid(hagrid_dir: Path, test_fraction: float, seed: int,
                 gestures: tuple[str, ...] = HAGRID_GESTURES) -> pd.DataFrame:
    """HaGRID v2 (Kapitanov et al.): the gestures that show the back of the hand.

    <hagrid_dir>/annotations/annotations/{train,val,test}/<gesture>.json and
    <hagrid_dir>/images/<gesture>/<uuid>.jpg. HaGRID's own train/val/test split is pooled; we
    make our own subject-disjoint split. ``user_id`` is already an anonymous hash.

    The returned frame has an extra column ``landmarks`` (JSON, normalised [x, y] of the 21
    annotated landmarks) that alignment needs; it is not written to the manifest CSV.
    Identities with fewer than 2 images are dropped (they cannot form a genuine pair).
    """
    dropped = {"no landmarks": 0, "image missing": 0}
    rows = []
    for split in ("train", "val", "test"):
        for gesture in gestures:
            rows += _hagrid_rows(hagrid_dir / "annotations" / "annotations" / split / f"{gesture}.json",
                                 gesture, hagrid_dir / "images", dropped)
    df = pd.DataFrame(rows)
    df["identity"] = df["subject"] + "_" + df["side"]
    n_images = df["identity"].map(df["identity"].value_counts())
    dropped["identity with 1 image"] = int((n_images < 2).sum())
    df = df[n_images >= 2].copy()
    print(f"[hagrid] kept {len(df)} images, dropped {dropped}")
    df["split"] = df["subject"].map(_subject_split(df["subject"].unique().tolist(), test_fraction, seed))
    return _finish(df, extra_columns=("age", "gender", "race", "landmarks"))
