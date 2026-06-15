from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import wandb

from .data import EMOTIONS
from .engine import TrainConfig, evaluate, fit, get_device
from .models import build_model, count_params

PROJECT = "ML_Asgn4"
ENTITY = None
WANDB_API_KEY = ""

RESULTS_CACHE = Path("results_cache.json")


def notebook_setup():
    sys.path.append(".")
    wandb.login(key=WANDB_API_KEY)


def init_run(architecture, run_name, config, job_type):
    return wandb.init(project=PROJECT, entity=ENTITY, group=architecture,
                      job_type=job_type, name=run_name, config=config, reinit=True)


def selection_score(val_acc, overfit_gap):
    return val_acc - 0.5 * max(0.0, overfit_gap - 0.05)


def cache_architecture_result(architecture, payload):
    cache = load_architecture_results()
    cache[architecture] = payload
    RESULTS_CACHE.write_text(json.dumps(cache, indent=2))


def load_architecture_results():
    if not RESULTS_CACHE.exists():
        return {}
    try:
        return json.loads(RESULTS_CACHE.read_text())
    except json.JSONDecodeError:
        return {}


def _hp_run_name(architecture, item):
    note = item.get("note", "run")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in note)
    return f"{architecture}_{safe}"


def _save_checkpoint(architecture, model_cfg, state_dict):
    path = Path("models")
    path.mkdir(exist_ok=True)
    ckpt = path / f"{architecture}_best.pt"
    torch.save({"state_dict": state_dict, "model_cfg": model_cfg,
                "architecture": architecture}, ckpt)


def run_training(architecture, run_name, job_type, model_cfg, train_cfg, loaders,
                 class_weights=None, log_cm=False, save_ckpt=False):
    device = get_device()
    config = {"model": architecture, **model_cfg, **asdict(train_cfg)}
    run = init_run(architecture, run_name, config, job_type)

    model = build_model(architecture, **model_cfg)
    wandb.summary["n_params"] = count_params(model)

    hist = fit(model, loaders, train_cfg, device=device, class_weights=class_weights,
               on_epoch_end=lambda ep, m: wandb.log(m, step=ep))

    score = selection_score(hist.best_val_acc, hist.overfit_gap)
    wandb.summary.update({
        "best_val_acc": hist.best_val_acc,
        "best_epoch": hist.best_epoch,
        "final_train_acc": hist.train_acc[-1],
        "overfit_gap": hist.overfit_gap,
        "selection_score": score,
    })

    if hist.best_state is not None:
        model.load_state_dict(hist.best_state)

    if log_cm:
        _, preds, targets = evaluate(model, loaders["val"], torch.nn.CrossEntropyLoss(),
                                     device, return_preds=True)
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            y_true=list(targets), preds=list(preds), class_names=EMOTIONS)})

    if save_ckpt and hist.best_state is not None:
        _save_checkpoint(architecture, model_cfg, hist.best_state)

    run.finish()
    return hist, {"best_val_acc": hist.best_val_acc, "overfit_gap": hist.overfit_gap,
                  "selection_score": score, "best_epoch": hist.best_epoch}


def run_hp_grid(architecture, grid, loaders, class_weights=None):
    rows, results = [], []
    for item in grid:
        model_cfg = item.get("model", {})
        train_cfg = TrainConfig(**item.get("train", {}))
        name = _hp_run_name(architecture, item)
        _, summ = run_training(architecture, name, "hp", model_cfg, train_cfg,
                               loaders, class_weights=class_weights)
        results.append((item, summ))
        rows.append({"note": item.get("note", ""), "run": name,
                     "val_acc": round(summ["best_val_acc"], 4),
                     "gap": round(summ["overfit_gap"], 4),
                     "score": round(summ["selection_score"], 4)})

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    print(df.to_string(index=False))
    best_item, best_summ = max(results, key=lambda r: r[1]["selection_score"])
    print(f"\nbest: {best_item.get('note', '')}  val_acc={best_summ['best_val_acc']:.4f}  "
          f"gap={best_summ['overfit_gap']:.4f}")
    return df, best_item, best_summ


def finish_run(architecture, best_item, loaders, final_train, class_weights=None, plot=None):
    from . import viz

    train_cfg = TrainConfig(**{**best_item["train"], **final_train})
    hist, summ = run_training(architecture, f"{architecture}_final", "final",
                              best_item["model"], train_cfg, loaders,
                              class_weights=class_weights, log_cm=True, save_ckpt=True)
    if plot:
        viz.plot_curves(hist, plot[0], plot[1])
    cache_architecture_result(architecture, {**summ, "model_cfg": best_item["model"],
                                               "train_cfg": best_item["train"]})
    print(summ)
    return hist, summ
