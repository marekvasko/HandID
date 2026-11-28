"""Data: image bank, train/test split, GPU augmentation and the Lightning DataModule.

Speed design (the CPU of the training machine is shared and slow):
    * every aligned crop is decoded ONCE into a uint8 array on NVMe (the "image bank",
      one memory-mapped .npy per dataset and input type), so workers never decode JPEGs,
    * augmentation = torchvision.transforms.v2 on uint8 tensors in the DataLoader workers;
      the expensive hue jitter is replaced by a cheap per-channel gain (``RandomWhiteBalance``),
    * conversion to float happens on the GPU; normalisation is part of the model, because
      CLIP / SigLIP / DINO use different statistics.

Which images are used for training is decided by the experiment *variant*:

    datasets   any of [ours, 11k, hd, hagrid]
    labels     identity  -> class = one hand of one person (needs person labels across sessions)
               session   -> class = one capture session of our data (no person labels needed;
                            two sessions of the same person are treated as different classes)

Test sets are always: every fold of our dataset not used for training, plus the
subject-disjoint test splits of 11k and HD (and of HaGRID, once its manifest exists).
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T

# ---------------------------------------------------------------------------------------------
# Image bank
# ---------------------------------------------------------------------------------------------
BANK_SIZE = 448  # = align.out_size


def _read(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((BANK_SIZE, BANK_SIZE)))


def load_bank(paths: list[str], cache_file: Path) -> np.ndarray:
    """uint8 array (N, H, W, 3) with the images in ``paths`` order, memory-mapped from disk."""
    # A bank is valid only for the exact list of images it was built from (row order matters).
    # A fingerprint of that list is stored next to it; a re-prepared manifest (e.g. a label fix
    # that renames 24 files but keeps the row count) triggers a rebuild.
    fingerprint = hashlib.sha1("\n".join(paths).encode()).hexdigest()
    fingerprint_file = cache_file.with_suffix(".paths.sha1")
    stored = fingerprint_file.read_text().strip() if fingerprint_file.exists() else None
    if cache_file.exists() and stored is not None and stored != fingerprint:
        print(f"{cache_file.name} was built from a different manifest, rebuilding")
        cache_file.unlink()
    if cache_file.exists() and len(np.load(cache_file, mmap_mode="r")) != len(paths):
        print(f"{cache_file.name} does not match its manifest (re-prepared?), rebuilding")
        cache_file.unlink()
    if not cache_file.exists():
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp.npy")
        bank = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(paths), BANK_SIZE, BANK_SIZE, 3))
        with ThreadPoolExecutor(16) as pool:
            for i, img in enumerate(pool.map(_read, paths)):
                bank[i] = img
        bank.flush()
        tmp.rename(cache_file)
        fingerprint_file.write_text(fingerprint)
    elif stored is None:  # bank built before fingerprints existed: adopt the current manifest
        fingerprint_file.write_text(fingerprint)
    return np.load(cache_file, mmap_mode="r")


def load_manifests(manifest_dir: str, bank_dir: str, input_name: str, aligned_dir: str
                   ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """Manifests (with a ``bank_row`` column) and the matching image banks."""
    manifests, banks = {}, {}
    for ds in ("ours", "11k", "hd", "hagrid"):
        if ds == "hagrid" and not (Path(manifest_dir) / "hagrid.csv").exists():
            continue  # optional dataset (scripts/prepare_data.py datasets=[hagrid])
        df = pd.read_csv(Path(manifest_dir) / f"{ds}.csv", dtype=str)
        if input_name == "masked":  # same crops, background removed (see align.hand_mask)
            df["path"] = df["path"].str.replace(aligned_dir, aligned_dir + "_masked", regex=False)
        df["bank_row"] = np.arange(len(df))
        manifests[ds] = df
        banks[ds] = load_bank(df["path"].tolist(), Path(bank_dir) / f"{input_name}_{ds}.npy")
    return manifests, banks


class BankImages(Dataset):
    """Rows of a manifest -> (transformed uint8 CHW image, class index, metadata targets).

    Class index is -1 for test data; metadata targets are -1 where unknown (see METADATA).
    """

    def __init__(self, df: pd.DataFrame, banks: dict[str, np.ndarray], transform: nn.Module,
                 labels: list[int] | None = None, metadata: np.ndarray | None = None):
        self.banks, self.transform = banks, transform
        self.rows = list(zip(df["dataset"], df["bank_row"]))
        self.labels = labels if labels is not None else [-1] * len(df)
        self.metadata = metadata if metadata is not None else np.full((len(df), len(METADATA)), -1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        ds, row = self.rows[i]
        image = torch.from_numpy(np.array(self.banks[ds][row])).permute(2, 0, 1)  # uint8 C x H x W
        return self.transform(image), self.labels[i], torch.from_numpy(self.metadata[i])



# ---------------------------------------------------------------------------------------------
# Metadata (only 11k Hands has it) - used as auxiliary prediction targets, see issue #5
# ---------------------------------------------------------------------------------------------
METADATA = {  # attribute -> class names; 11k ages are mostly 21-22, hence the narrow bins
    "gender": ["female", "male"],
    "skinColor": ["very fair", "fair", "medium", "dark"],
    "accessories": ["0", "1"],
    "nailPolish": ["0", "1"],
    "age": ["<=20", "21", "22", ">=23"],
}


def _age_bin(age: str) -> str:
    a = int(age)
    return "<=20" if a <= 20 else ">=23" if a >= 23 else str(a)


def metadata_targets(df: pd.DataFrame, handinfo_csv: str) -> np.ndarray:
    """(N, len(METADATA)) int64 class indices; -1 for rows without metadata (ours, hd)."""
    info = pd.read_csv(handinfo_csv, dtype=str).set_index("imageName")
    info["age"] = info["age"].map(_age_bin)
    out = np.full((len(df), len(METADATA)), -1, dtype=np.int64)
    is_11k = (df["dataset"] == "11k").to_numpy()
    rows = info.loc[df.loc[is_11k, "src_path"].map(lambda p: Path(p).name)]
    for k, (attr, classes) in enumerate(METADATA.items()):
        out[is_11k, k] = rows[attr].map({c: i for i, c in enumerate(classes)}).fillna(-1).to_numpy()
    return out


# ---------------------------------------------------------------------------------------------
# Transforms (uint8 C x H x W in, uint8 C x S x S out; run in the DataLoader workers)
# ---------------------------------------------------------------------------------------------
class RandomWhiteBalance(nn.Module):
    """Multiply each colour channel by its own random gain - a camera white-balance change.

    Built from basic tensor ops because torchvision has no such transform; it is also much
    cheaper than ColorJitter's hue shift (which needs an RGB <-> HSV round trip).
    """

    def __init__(self, max_gain: float = 0.1, p: float = 0.5):
        super().__init__()
        self.max_gain, self.p = max_gain, p

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) > self.p:
            return image
        gains = 1 + (torch.rand(3, 1, 1) * 2 - 1) * self.max_gain
        return (image.float() * gains).clamp_(0, 255).to(torch.uint8)


def train_transform(size: int) -> T.Compose:
    # no horizontal flip: all hands are canonicalised to look like right hands
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.85, 1.15), antialias=True),
        T.RandomRotation(15),
        T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)], p=0.9),
        RandomWhiteBalance(max_gain=0.1, p=0.5),
        T.RandomGrayscale(p=0.1),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
    ])


def eval_transform(size: int) -> T.Resize:
    return T.Resize((size, size), antialias=True)


def to_float(images: torch.Tensor) -> torch.Tensor:
    """uint8 batch (on the GPU) -> float in [0, 1]."""
    return images.float().div_(255)


# ---------------------------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------------------------
def split_train_test(manifests: dict[str, pd.DataFrame], variant_datasets: list[str], fold: int | None,
                     labels: str = "identity", uses_hagrid: bool = False
                     ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return (training rows with a ``label`` column, {test set name: rows})."""
    ours = manifests["ours"]
    held_out = f"fold_{fold}" if fold is not None else None

    train_parts = []
    if "ours" in variant_datasets:
        train_parts.append(ours[ours.split != held_out])
    for ds in ("11k", "hd", "hagrid"):
        if ds in variant_datasets:
            train_parts.append(manifests[ds][manifests[ds].split == "train"])
    train = (pd.concat(train_parts) if train_parts else ours.iloc[:0]).reset_index(drop=True)

    train["label"] = train["identity"]
    if labels == "session":  # our data: every session is its own class, person links are ignored
        is_ours = train.dataset == "ours"
        train.loc[is_ours, "label"] = train.loc[is_ours, "identity"] + "/s" + train.loc[is_ours, "session"]

    test = {}
    used_folds = set(train.loc[train.dataset == "ours", "split"])
    for f in sorted(set(ours.split) - used_folds, key=lambda s: int(s.split("_")[1])):
        test[f"ours/{f}"] = ours[ours.split == f]
    for ds in ("11k", "hd"):
        test[f"{ds}/test"] = manifests[ds][manifests[ds].split == "test"]
    # HaGRID's test split (7k images, 27 M pairs) is only evaluated for runs that use HaGRID, so the
    # pre-defined test configuration of all other experiments stays unchanged (issue #8).
    if "hagrid" in manifests and uses_hagrid:
        test["hagrid/test"] = manifests["hagrid"][manifests["hagrid"].split == "test"]
    return train, test


# ---------------------------------------------------------------------------------------------
# Lightning DataModule
# ---------------------------------------------------------------------------------------------
class HandDataModule(L.LightningDataModule):
    def __init__(self, train_df: pd.DataFrame, banks: dict[str, np.ndarray], image_size: int, batch_size: int,
                 num_workers: int, balance_power: float, metadata: np.ndarray | None = None):
        super().__init__()
        self.train_df, self.banks, self.metadata = train_df, banks, metadata
        self.classes = sorted(train_df["label"].unique())
        self.class_index = {c: i for i, c in enumerate(self.classes)}
        self.image_size, self.batch_size, self.num_workers = image_size, batch_size, num_workers
        self.balance_power = balance_power

    def train_dataloader(self) -> DataLoader:
        labels = [self.class_index[c] for c in self.train_df["label"]]
        ds = BankImages(self.train_df, self.banks, train_transform(self.image_size), labels, self.metadata)
        # Each image is drawn with probability ~ 1 / n_images(class)^power.
        # power=0: plain shuffling, power=1: every class equally often. 0.5 is a compromise
        # that keeps heavily recorded hands from dominating without over-sampling tiny ones.
        n = self.train_df["label"].map(self.train_df["label"].value_counts()).to_numpy(float)
        sampler = WeightedRandomSampler(torch.as_tensor(n ** -self.balance_power), num_samples=len(ds), replacement=True)
        return DataLoader(ds, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers,
                          pin_memory=True, drop_last=True, persistent_workers=True)

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        images, labels, metadata = batch
        return to_float(images), labels, metadata


def eval_loader(df: pd.DataFrame, banks: dict[str, np.ndarray], image_size: int, batch_size: int,
                num_workers: int) -> DataLoader:
    return DataLoader(BankImages(df, banks, eval_transform(image_size)), batch_size=batch_size,
                      num_workers=num_workers, pin_memory=True)


def describe(train: pd.DataFrame, test: dict[str, pd.DataFrame]) -> str:
    lines = [f"train: {len(train)} images, {train['label'].nunique() if len(train) else 0} classes "
             f"({', '.join(f'{k}={v}' for k, v in train.dataset.value_counts().items())})"]
    lines += [f"test {k}: {len(v)} images, {v.identity.nunique()} identities" for k, v in test.items()]
    return "\n".join(lines)
