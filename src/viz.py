from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .data import EMOTIONS

PLOTS = Path("plots")


def _save(fig, name: str) -> Path:
    PLOTS.mkdir(exist_ok=True)
    path = PLOTS / name
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def plot_curves(history, title: str, name: str) -> Path:
    epochs = range(1, len(history.train_loss) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history.train_loss, label="train")
    ax1.plot(epochs, history.val_loss, label="val")
    ax1.set_title("loss")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax2.plot(epochs, history.train_acc, label="train")
    ax2.plot(epochs, history.val_acc, label="val")
    ax2.set_title("acc")
    ax2.set_xlabel("epoch")
    ax2.legend()
    fig.suptitle(title)
    return _save(fig, name)


def plot_class_balance(labels, name: str = "01_class_balance.png") -> Path:
    counts = np.bincount(labels, minlength=len(EMOTIONS))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(EMOTIONS, counts)
    ax.set_title("emotion counts")
    ax.set_ylabel("count")
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha="right")
    return _save(fig, name)


def plot_sample_faces(images, labels, name: str = "02_sample_faces.png", n: int = 14) -> Path:
    fig, axes = plt.subplots(2, 7, figsize=(12, 4))
    for ax, idx in zip(axes.ravel(), range(n)):
        ax.imshow(images[idx], cmap="gray")
        ax.set_title(EMOTIONS[int(labels[idx])], fontsize=8)
        ax.axis("off")
    return _save(fig, name)
