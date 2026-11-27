"""Datasets and the Lightning DataModule.

Which images are used for training is decided by the experiment *variant*:

    ours_only       our dataset, all folds except the held-out ``fold``
    external_only   11k + HD train splits
    ours_external   both of the above
    zero_shot       nothing (pretrained backbone is evaluated directly)

Test sets are always: every fold of our dataset not used for training, plus the
subject-disjoint test splits of 11k and HD.
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def train_transform(size: int) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.85, 1.15), antialias=True),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
        T.RandomGrayscale(p=0.1),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        # no horizontal flip: all hands are canonicalised to look like right hands
    ])


def eval_transform(size: int) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.Resize(size, antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class HandImages(Dataset):
    """Rows of a manifest -> (image tensor, class index). Class index is -1 for test data."""

    def __init__(self, df: pd.DataFrame, transform, class_index: dict[str, int] | None = None):
        self.paths = df["path"].tolist()
        self.labels = [class_index[i] for i in df["identity"]] if class_index else [-1] * len(df)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def load_manifests(manifest_dir: str) -> dict[str, pd.DataFrame]:
    return {ds: pd.read_csv(Path(manifest_dir) / f"{ds}.csv", dtype=str) for ds in ("ours", "11k", "hd")}


def split_train_test(manifests: dict[str, pd.DataFrame], variant_datasets: list[str], fold: int | None
                     ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return (training rows, {test set name: rows})."""
    ours = manifests["ours"]
    held_out = f"fold_{fold}" if fold is not None else None

    train_parts = []
    if "ours" in variant_datasets:
        train_parts.append(ours[ours.split != held_out])
    for ds in ("11k", "hd"):
        if ds in variant_datasets:
            train_parts.append(manifests[ds][manifests[ds].split == "train"])
    train = pd.concat(train_parts) if train_parts else ours.iloc[:0]

    test = {}
    used_folds = set(train.loc[train.dataset == "ours", "split"])
    for f in sorted(set(ours.split) - used_folds, key=lambda s: int(s.split("_")[1])):
        test[f"ours/{f}"] = ours[ours.split == f]
    for ds in ("11k", "hd"):
        test[f"{ds}/test"] = manifests[ds][manifests[ds].split == "test"]
    return train.reset_index(drop=True), test


class HandDataModule(L.LightningDataModule):
    def __init__(self, train_df: pd.DataFrame, image_size: int, batch_size: int, num_workers: int,
                 balance_power: float):
        super().__init__()
        self.train_df = train_df
        self.classes = sorted(train_df["identity"].unique())
        self.class_index = {c: i for i, c in enumerate(self.classes)}
        self.image_size, self.batch_size, self.num_workers = image_size, batch_size, num_workers
        self.balance_power = balance_power

    def train_dataloader(self) -> DataLoader:
        ds = HandImages(self.train_df, train_transform(self.image_size), self.class_index)
        # Each image is drawn with probability ~ 1 / n_images(identity)^power.
        # power=0: plain shuffling, power=1: every identity equally often. 0.5 is a compromise
        # that keeps heavily recorded hands from dominating without over-sampling tiny ones.
        n = self.train_df["identity"].map(self.train_df["identity"].value_counts()).to_numpy(float)
        weights = torch.as_tensor(n ** -self.balance_power)
        sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
        return DataLoader(ds, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers,
                          pin_memory=True, drop_last=True, persistent_workers=True)


def eval_loader(df: pd.DataFrame, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(HandImages(df, eval_transform(image_size)), batch_size=batch_size,
                      num_workers=num_workers, pin_memory=True)


def describe(train: pd.DataFrame, test: dict[str, pd.DataFrame]) -> str:
    lines = [f"train: {len(train)} images, {train.identity.nunique()} identities "
             f"({', '.join(f'{k}={v}' for k, v in train.dataset.value_counts().items())})"]
    lines += [f"test {k}: {len(v)} images, {v.identity.nunique()} identities" for k, v in test.items()]
    return "\n".join(lines)

