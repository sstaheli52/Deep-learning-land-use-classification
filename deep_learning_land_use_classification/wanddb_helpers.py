import wandb
import torch
import numpy as np
from deep_learning_land_use_classification import config

PROJECT_SINGLE = "DL_single-label-land-use-classification"
PROJECT_MULTI  = "DL_multi-label-land-use-classification"
ENTITY         = "sstaheli52-wageningen-university-and-research"

def init_run(
    task: str,
    architecture: str,
    num_classes: int,
    loss: str,
    epochs: int          = 5,
    batch_size: int      = 32,
    learning_rate: float = 1e-4,
    optimizer: str       = "Adam",
    pretrained: bool     = True,
    pretraining_dataset: str = "ImageNetV2",
    pretraining_source: str  = "torchvision",
    weights: str         = "IMAGENET1K_V2",
    model_name: str      = None,
    augmentation: bool   = False,
    early_stopping: bool = False,
    patience: int        = 2,
    min_delta: float     = 0.001,
    run_name: str        = None,
    extra_config: dict   = None,
    backbone_frozen: bool = False,
    backbone_learning_rate: float = 1e-4,
    dropout: float = None,
    
) -> wandb.sdk.wandb_run.Run:
    assert task in ("single", "multi"), "task must be 'single' or 'multi'"

    project = PROJECT_SINGLE if task == "single" else PROJECT_MULTI

    base_config = {
        "task":                 task,
        "architecture":         architecture,
        "model_name":           model_name,
        "pretrained":           pretrained,
        "pretraining_dataset":  pretraining_dataset,
        "pretraining_source":   pretraining_source,
        "weights":              weights,
        "num_classes":          num_classes,
        "augmentation":         augmentation,
        "epochs":               epochs,
        "batch_size":           batch_size,
        "learning_rate":        learning_rate,
        "optimizer":            optimizer,
        "loss":                 loss,
        "early_stopping":       early_stopping,
        "patience":             patience if early_stopping else None,
        "min_delta":            min_delta if early_stopping else None,
        "backbone_frozen":       backbone_frozen,
        "backbone_learning_rate": backbone_learning_rate, 
        "dropout":              dropout,
    }

    if extra_config:
        base_config.update(extra_config)

    run = wandb.init(
        entity=ENTITY,
        project=project,
        name=run_name,
        dir=config.WANDB_DIR,
        config=base_config,
    )
    return run


# ── Metric logging ────────────────────────────────────────────────────────────

def log_epoch(
    run: wandb.sdk.wandb_run.Run,
    epoch: int,
    train_loss: float,
    val_loss: float,
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    p_macro: float,
    r_macro: float,
    f1_macro: float,
    class_names: list[str],
) -> None:
    metrics = {
        "epoch":            epoch,
        "train_loss":       train_loss,
        "val_loss":        val_loss,
        "macro/precision":  p_macro,
        "macro/recall":     r_macro,
        "macro/f1":         f1_macro,
    }

    for i, name in enumerate(class_names):
        safe = _safe_name(name)
        metrics[f"class/{safe}/precision"] = float(precision[i])
        metrics[f"class/{safe}/recall"]    = float(recall[i])
        metrics[f"class/{safe}/f1"]        = float(f1[i])

    table = wandb.Table(columns=["class", "precision", "recall", "f1"])
    for i, name in enumerate(class_names):
        table.add_data(name, float(precision[i]), float(recall[i]), float(f1[i]))
    metrics["class_metrics_table"] = table

    run.log(metrics)


def log_model_summary(
    run: wandb.sdk.wandb_run.Run,
    model: torch.nn.Module,
    log: str     = "parameters",
    log_freq: int = 100,
) -> None:
    run.watch(model, log=log, log_freq=log_freq)

# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_name(class_name: str) -> str:
    return class_name.replace(" ", "_").replace("-", "_")