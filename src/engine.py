from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    scheduler: str = "none"
    label_smoothing: float = 0.0
    patience: int = 0


def make_scheduler(optimizer, cfg: TrainConfig):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    return None


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        n += x.size(0)
        if return_preds:
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    metrics = {"loss": total_loss / n, "acc": correct / n}
    if return_preds:
        return metrics, np.concatenate(all_preds), np.concatenate(all_targets)
    return metrics


@dataclass
class History:
    train_loss: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = 0
    best_state: dict | None = None
    overfit_gap: float = 0.0


def fit(model, loaders, cfg: TrainConfig, device=None,
        class_weights=None, on_epoch_end=None) -> History:
    device = device or get_device()
    model = model.to(device)

    weight = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(optimizer, cfg)

    hist = History()
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val = evaluate(model, loaders["val"], criterion, device)
        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        hist.train_loss.append(tr_loss); hist.train_acc.append(tr_acc)
        hist.val_loss.append(val["loss"]); hist.val_acc.append(val["acc"])

        metrics = {"train_loss": tr_loss, "train_acc": tr_acc,
                   "val_loss": val["loss"], "val_acc": val["acc"], "lr": lr}
        if on_epoch_end is not None:
            on_epoch_end(epoch, metrics)
        print(f"epoch {epoch:3d}  train_acc {tr_acc:.4f}  val_acc {val['acc']:.4f}  "
              f"train_loss {tr_loss:.4f}  val_loss {val['loss']:.4f}")

        if val["acc"] > hist.best_val_acc:
            hist.best_val_acc = val["acc"]
            hist.best_epoch = epoch
            hist.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if cfg.patience and epochs_no_improve >= cfg.patience:
                print(f"stopped early epoch {epoch}")
                break

    hist.overfit_gap = (hist.train_acc[hist.best_epoch - 1] - hist.best_val_acc)
    return hist
