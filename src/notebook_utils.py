from __future__ import annotations

import torch.nn as nn

from .data import NUM_CLASSES, class_weights, make_loaders
from .engine import TrainConfig, get_device
from .models import build_model
from . import wandb_utils


def setup():
    wandb_utils.notebook_setup()


def quick_check(arch, model_cfg, loaders):
    device = get_device()
    model = build_model(arch, **model_cfg).to(device)
    x, y = next(iter(loaders["train"]))
    x, y = x.to(device), y.to(device)
    out = model(x)
    assert out.shape == (x.size(0), NUM_CLASSES)
    loss = nn.CrossEntropyLoss()(out, y).item()
    print(f"{arch}: output {tuple(out.shape)}, loss={loss:.3f}")


def run_architecture(
    arch,
    grid,
    final_train,
    plot,
    augment=True,
    class_weights_on=False,
    ablation=None,
):
    loaders = make_loaders(augment=augment)
    loaders_noaug = make_loaders(augment=False) if ablation else None
    cw = class_weights(loaders["train"]) if class_weights_on else None

    quick_check(arch, grid[0]["model"], loaders)

    table, best_item, best_summ = wandb_utils.run_hp_grid(
        arch, grid, loaders, class_weights=cw,
    )

    if ablation:
        abl_model = ablation["model"]
        abl_train = TrainConfig(**ablation.get("train", {"epochs": 25, "lr": 1e-3}))
        wandb_utils.run_training(
            arch, f"{arch}_ablation_noaug", "hp", abl_model, abl_train,
            loaders_noaug, class_weights=cw,
        )
        wandb_utils.run_training(
            arch, f"{arch}_ablation_aug", "hp", abl_model, abl_train,
            loaders, class_weights=cw,
        )

    wandb_utils.finish_run(
        arch, best_item, loaders,
        final_train=final_train,
        class_weights=cw,
        plot=plot,
    )
    return table, best_item, best_summ
