from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import BATCH_SIZE, DATA_DIR, NUM_WORKERS

IMG_SIZE = 48
NUM_CLASSES = 7
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def load_fer(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def decode_pixels(pixel_series: pd.Series) -> np.ndarray:
    arr = np.stack([np.array(p.split(), dtype=np.uint8) for p in pixel_series])
    return arr.reshape(-1, IMG_SIZE, IMG_SIZE)


class FERDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray | None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.images[idx]).float().div(255.0).unsqueeze(0)
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is None:
            return img
        return img, int(self.labels[idx])


def build_transforms(augment: bool):
    norm = transforms.Normalize(mean=[0.5], std=[0.5])
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomCrop(IMG_SIZE, padding=4),
            norm,
        ])
    return transforms.Compose([norm])


def make_loaders(data_dir: str | Path = DATA_DIR, batch_size: int = BATCH_SIZE,
                 augment: bool = True, num_workers: int = NUM_WORKERS):
    df = load_fer(Path(data_dir) / "icml_face_data.csv")
    splits = {
        "train": df[df["Usage"] == "Training"],
        "val": df[df["Usage"] == "PublicTest"],
        "test": df[df["Usage"] == "PrivateTest"],
    }

    loaders = {}
    for name, part in splits.items():
        images = decode_pixels(part["pixels"])
        labels = part["emotion"].to_numpy()
        tfm = build_transforms(augment=(augment and name == "train"))
        ds = FERDataset(images, labels, transform=tfm)
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=(name == "train"),
            num_workers=num_workers, pin_memory=True, drop_last=(name == "train"),
        )
    return loaders


def make_kaggle_test_loader(data_dir: str | Path = DATA_DIR, batch_size: int = BATCH_SIZE,
                            num_workers: int = NUM_WORKERS) -> DataLoader:
    df = load_fer(Path(data_dir) / "test.csv")
    images = decode_pixels(df["pixels"])
    ds = FERDataset(images, None, transform=build_transforms(augment=False))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def class_weights(loader: DataLoader) -> torch.Tensor:
    labels = np.array(loader.dataset.labels)
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)
