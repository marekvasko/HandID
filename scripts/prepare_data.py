"""Step 1: prepare all data (one-off, ~15 min).

    uv run scripts/prepare_data.py

Produces in ``paths.data_dir``:
    external/            unpacked 11k Hands + HD archives
    manifests/<ds>.csv   one row per image (see src/handid/manifests.py)
    aligned/<ds>/...     canonical hand crops (see src/handid/align.py)
    aligned_masked/...   the same crops with the background blacked out (input=masked)
"""

import zipfile
from multiprocessing import Pool
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from handid import align, manifests

HANDINFO_URL = "https://drive.usercontent.google.com/download?id=1RC86-rVOR8c93XAfM9b9R45L7C2B0FdA&export=download"


def unpack_external(archives: Path, out: Path) -> None:
    if not (out / "11k" / "Hands").exists():
        print("unpacking 11k Hands ...")
        zipfile.ZipFile(archives / "Hands.zip").extractall(out / "11k")
    if not (out / "11k" / "HandInfo.csv").exists():
        # The metadata (subject id, dorsal/palmar) is not inside the zip.
        import urllib.request
        urllib.request.urlretrieve(HANDINFO_URL, out / "11k" / "HandInfo.csv")
    if not (out / "hd" / "HD").exists():
        print("unpacking HD (RAR archive despite the .7z name) ...")
        import libarchive
        (out / "hd").mkdir(parents=True, exist_ok=True)
        import os
        cwd = os.getcwd()
        os.chdir(out / "hd")
        libarchive.extract_file(str(archives / "HD.7z"))
        os.chdir(cwd)


def align_manifest(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Align every image of a manifest, add columns ``path`` (aligned crop) and ``align``."""
    out_root = Path(cfg.paths.aligned_dir)
    df = df.copy()
    df["path"] = [str(out_root / r.dataset / r.identity / f"{i:05d}.jpg") for i, r in enumerate(df.itertuples())]
    todo = [(r.src_path, r.path, r.dataset, r.side) for r in df.itertuples() if not Path(r.path).exists()]
    methods = {}
    if todo:
        init_args = (cfg.paths.hand_landmarker, cfg.align.out_size, cfg.align.margin)
        with Pool(cfg.align.workers, initializer=align.init_worker, initargs=init_args) as pool:
            results = pool.imap(align.align_one, todo, chunksize=4)
            for job, method in tqdm(zip(todo, results), total=len(todo), desc=f"align {df.dataset.iloc[0]}"):
                methods[job[1]] = method
    # keep earlier results for images aligned in a previous run
    old = Path(cfg.paths.manifest_dir) / f"{df.dataset.iloc[0]}.csv"
    if old.exists():
        prev = pd.read_csv(old, dtype=str)
        if "align" in prev:
            methods = {**dict(zip(prev["path"], prev["align"])), **methods}
    df["align"] = df["path"].map(methods).fillna("unknown")
    n_err = df["align"].str.startswith("error").sum()
    if n_err:
        print(f"dropping {n_err} unreadable images")
        df = df[~df["align"].str.startswith("error")]
    return df


def mask_manifest(df: pd.DataFrame, cfg: DictConfig) -> None:
    """Write a background-masked copy of every aligned crop to ``<aligned_dir>_masked``."""
    src_root, dst_root = cfg.paths.aligned_dir, cfg.paths.aligned_dir + "_masked"
    todo = [(p, p.replace(src_root, dst_root, 1)) for p in df["path"]]
    todo = [job for job in todo if not Path(job[1]).exists()]
    if not todo:
        return
    init_args = (cfg.paths.hand_landmarker, cfg.align.out_size, cfg.align.margin)
    with Pool(cfg.align.workers, initializer=align.init_worker, initargs=init_args) as pool:
        results = list(tqdm(pool.imap(align.mask_one, todo, chunksize=8), total=len(todo),
                            desc=f"mask {df.dataset.iloc[0]}"))
    print(f"[{df.dataset.iloc[0]}] masked {results.count('masked') / len(results) * 100:.1f} % "
          "(the rest: no hand found in the crop, kept unmasked)")


@hydra.main(config_path="../configs", config_name="prepare", version_base="1.3")
def main(cfg: DictConfig) -> None:
    unpack_external(Path(cfg.paths.external_archives), Path(cfg.paths.external_dir))
    builders = {
        "ours": lambda: manifests.build_ours(Path(cfg.paths.ours_root), Path(cfg.paths.private_dir)),
        "11k": lambda: manifests.build_11k(Path(cfg.paths.external_dir), cfg.external_test_fraction, cfg.seed),
        "hd": lambda: manifests.build_hd(Path(cfg.paths.external_dir), cfg.external_test_fraction, cfg.seed),
    }
    out_dir = Path(cfg.paths.manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, build in builders.items():
        df = align_manifest(build(), cfg)
        df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"[{name}] {len(df)} images, {df.subject.nunique()} subjects, {df.identity.nunique()} identities")
        print(df["align"].value_counts(normalize=True).round(3).to_string())
        mask_manifest(df, cfg)


if __name__ == "__main__":
    main()
