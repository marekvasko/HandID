"""Data: image bank, train/test split, GPU augmentation and the Lightning DataModule.

Speed design (the CPU of the training machine is shared and slow, the GPU is fast):
    * every aligned crop is decoded ONCE into a uint8 array on NVMe (the "image bank",
      one memory-mapped .npy per dataset and input type); loading a batch is a memcpy,
    * all augmentation runs on the GPU, batched, with kornia (``GpuAugment``),
    * normalisation is part of the model, because CLIP / SigLIP / DINO use different statistics.

Which images are used for training is decided by the experiment *variant*:

    datasets   any of [ours, 11k, hd]
    labels     identity  -> class = one hand of one person (needs person labels across sessions)
               session   -> class = one capture session of our data (no person labels needed;
                            two sessions of the same person are treated as different classes)

Test sets are always: every fold of our dataset not used for training, plus the
subject-disjoint test splits of 11k and HD.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import kornia.augmentation as K
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ---------------------------------------------------------------------------------------------
# Image bank
# ---------------------------------------------------------------------------------------------
BANK_SIZE = 448  # = align.out_size


def _read(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((BANK_SIZE, BANK_SIZE)))


def load_bank(paths: list[str], cache_file: Path) -> np.ndarray:
    """uint8 array (N, H, W, 3) with the images in ``paths`` order, memory-mapped from disk."""
    if not cache_file.exists():
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp.npy")
        bank = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(paths), BANK_SIZE, BANK_SIZE, 3))
        with ThreadPoolExecutor(16) as pool:
            for i, img in enumerate(pool.map(_read, paths)):
                bank[i] = img
        bank.flush()
        tmp.rename(cache_file)
    return np.load(cache_file, mmap_mode="r")


def load_manifests(manifest_dir: str, bank_dir: str, input_name: str, aligned_dir: str
                   ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """Manifests (with a ``bank_row`` column) and the matching image banks."""
    manifests, banks = {}, {}
    for ds in ("ours", "11k", "hd"):
        df = pd.read_csv(Path(manifest_dir) / f"{ds}.csv", dtype=str)
        if input_name == "masked":  # same crops, background removed (see align.hand_mask)
            df["path"] = df["path"].str.replace(aligned_dir, aligned_dir + "_masked", regex=False)
        df["bank_row"] = np.arange(len(df))
        manifests[ds] = df
        banks[ds] = load_bank(df["path"].tolist(), Path(bank_dir) / f"{input_name}_{ds}.npy")
    return manifests, banks


class BankImages(Dataset):
    """Rows of a manifest -> (uint8 HWC image, class index, metadata targets).

    Class index is -1 for test data; metadata targets are -1 where unknown (see METADATA).
    """

    def __init__(self, df: pd.DataFrame, banks: dict[str, np.ndarray], labels: list[int] | None = None,
                 metadata: np.ndarray | None = None):
        self.banks = banks
        self.rows = list(zip(df["dataset"], df["bank_row"]))
        self.labels = labels if labels is not None else [-1] * len(df)
        self.metadata = metadata if metadata is not None else np.full((len(df), len(METADATA)), -1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        ds, row = self.rows[i]
        return torch.from_numpy(np.array(self.banks[ds][row])), self.labels[i], torch.from_numpy(self.metadata[i])


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
# GPU transforms (input: uint8 B x H x W x 3 on the GPU, output: float B x 3 x S x S in [0, 1])
# ---------------------------------------------------------------------------------------------
def to_float(images: torch.Tensor) -> torch.Tensor:
    return images.permute(0, 3, 1, 2).float().div_(255)


class GpuAugment(nn.Module):
    """Per-sample random augmentation of a whole batch on the GPU.

    No horizontal flip: all hands are canonicalised to look like right hands.
    """

    def __init__(self, size: int):
        super().__init__()
        self.augment = K.AugmentationSequential(
            K.RandomResizedCrop((size, size), scale=(0.6, 1.0), ratio=(0.85, 1.15)),
            K.RandomRotation(15.0, p=1.0),
            K.ColorJitter(0.3, 0.3, 0.3, 0.03, p=0.9),
            K.RandomGrayscale(p=0.1),
            K.RandomGaussianBlur((5, 5), (0.1, 2.0), p=0.2),
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.augment(to_float(images)).clamp_(0, 1)


@torch.no_grad()
def eval_transform(images: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(to_float(images), size=(size, size), mode="bilinear", antialias=True, align_corners=False)


# ---------------------------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------------------------
def split_train_test(manifests: dict[str, pd.DataFrame], variant_datasets: list[str], fold: int | None,
                     labels: str = "identity") -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return (training rows with a ``label`` column, {test set name: rows})."""
    ours = manifests["ours"]
    held_out = f"fold_{fold}" if fold is not None else None

    train_parts = []
    if "ours" in variant_datasets:
        train_parts.append(ours[ours.split != held_out])
    for ds in ("11k", "hd"):
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
        self.batch_size, self.num_workers, self.balance_power = batch_size, num_workers, balance_power
        self.augment = GpuAugment(image_size)

    def train_dataloader(self) -> DataLoader:
        labels = [self.class_index[c] for c in self.train_df["label"]]
        ds = BankImages(self.train_df, self.banks, labels, self.metadata)
        # Each image is drawn with probability ~ 1 / n_images(class)^power.
        # power=0: plain shuffling, power=1: every class equally often. 0.5 is a compromise
        # that keeps heavily recorded hands from dominating without over-sampling tiny ones.
        n = self.train_df["label"].map(self.train_df["label"].value_counts()).to_numpy(float)
        sampler = WeightedRandomSampler(torch.as_tensor(n ** -self.balance_power), num_samples=len(ds), replacement=True)
        return DataLoader(ds, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers,
                          pin_memory=True, drop_last=True, persistent_workers=True)

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        images, labels, metadata = batch
        if self.trainer.training:
            images = self.augment(images)
        return images, labels, metadata


def eval_loader(df: pd.DataFrame, banks: dict[str, np.ndarray], batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(BankImages(df, banks), batch_size=batch_size, num_workers=num_workers, pin_memory=True)


def describe(train: pd.DataFrame, test: dict[str, pd.DataFrame]) -> str:
    lines = [f"train: {len(train)} images, {train['label'].nunique() if len(train) else 0} classes "
             f"({', '.join(f'{k}={v}' for k, v in train.dataset.value_counts().items())})"]
    lines += [f"test {k}: {len(v)} images, {v.identity.nunique()} identities" for k, v in test.items()]
    return "\n".join(lines)
