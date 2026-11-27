"""Build one manifest CSV per dataset.

Every manifest has the same columns, so the rest of the code never needs to know
which dataset an image came from:

    dataset     ours | 11k | hd
    subject     anonymous subject id, e.g. "ours_007", "11k_0000123", "hd_0042"
    side        left | right        (which hand)
    identity    f"{subject}_{side}" (the class we learn; left and right hands are different biometrics)
    session     capture session id (string). External datasets have a single session "0".
    split       ours: "fold_0".."fold_9" (people-disjoint folds, given)
                external: "train" | "test" (subject-disjoint, fixed seed)
    src_path    absolute path of the original image

PRIVACY: the real names in our dataset are replaced by ``ours_XXX``. The mapping is
written to ``paths.private_dir`` (gitignored) and must never be committed or logged.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pandas as pd
import yaml

COLUMNS = ["dataset", "subject", "side", "identity", "session", "split", "src_path"]


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df["identity"] = df["subject"] + "_" + df["side"]
    return df[COLUMNS].sort_values(["subject", "side", "session", "src_path"]).reset_index(drop=True)


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
