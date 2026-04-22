# Deep-learning-land-use-classification

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Authors: Sophia Staheli & Tomer Peled

WUR MGI project for land-use image classification on the UC Merced satellite image dataset.
This repository contains the code, notebooks, saved models, and final report for comparing
ResNet50 and Vision Transformer (ViT) approaches on both single-label and multi-label tasks.

For the full experimental write-up, figures, per-class metrics, and discussion, see the project
report in [Reports/DL_Project_Sophia_Tomer.pdf](Reports/DL_Project_Sophia_Tomer.pdf).

## Overview

The project studies the UC Merced Land Use
Dataset. Two model families were compared throughout the work:

- A pretrained ResNet50 baseline, chosen as a strong convolutional reference model.
- A pretrained ViT-L model initialized with DINOv3 satellite pretraining, used as the main
    transformer-based alternative.

The report focuses on how architecture choice, backbone freezing, augmentation, dropout, and
learning rate affect performance. The code in this repository mirrors
those experiments and provides reusable helpers for loading data, splitting the dataset,
tracking metrics, and applying early stopping.

### Key findings

The report highlights a clear split between the two tasks:

| Task | Best model | Validation F1 | Test F1 | Notes |
| --- | --- | --- | --- | --- |
| Single-label | ResNet50 | 0.99 | 0.9881 | Near-perfect performance on the 21-class benchmark |
| Multi-label | ViT with the last transformer block unfrozen | 0.94 | 0.9315 | Best overall multi-label result |
| Multi-label baseline | Frozen ViT variants | 0.81-0.85 | - | Lower precision, showing limited domain adaptation |

The main interpretation is that the single-label benchmark is largely saturated, while the
multi-label setup is more informative for model comparison. On the latter, partial fine-tuning of
the ViT backbone matters more than extra regularization or augmentation alone.

### Data and training setup

The project uses a 64/16/20 train/validation/test split. The data pipeline resizes images to
224x224, uses cross-entropy for single-label experiments, uses BCEWithLogitsLoss for multi-label
experiments, and relies on AdamW optimization with early stopping. W&B is used to log training
metrics and model summaries.

The package code centralizes the shared project paths in `deep_learning_land_use_classification`
and includes small utility modules for:

- downloading and splitting the dataset,
- computing classification metrics and confusion matrices,
- stopping training early when validation loss stalls,
- logging runs and per-class metrics to Weights & Biases.

## Get started

1. Create a new Python 3.10 environment.
2. Install the package in editable mode with `pip install -e .`.
3. If you want to reproduce the dataset download and local raw-data layout, run `make data`.
4. To use Weights & Biases, activate your environment and follow the steps in the
    [W&B quickstart](https://docs.wandb.ai/models/quickstart#command-line), or run `wandb login`.


Most exploratory work lives in `notebooks/`. The notebook sequence covers the single-label and
multi-label data exploration, the ResNet50 and ViT experiments, and augmentation
variants. The saved model checkpoints are stored in `models/`, and the final report is in
`Reports/`.

## Project organisation

```
├── LICENSE
├── Makefile                    <- Convenience commands such as make data, make lint, and make format
├── README.md                   <- Project overview and usage notes
├── Reports/
│   └── DL_Project_Sophia_Tomer.pdf
├── data/
│   ├── external/               <- Third-party or staged source data
│   ├── interim/                <- Intermediate transformed data
│   ├── processed/              <- Final training and evaluation data
│   └── raw/                    <- Original dataset assets
│       ├── Images/             <- UC Merced image folders
│       └── LandUse_Multilabeled.txt
├── deep_learning_land_use_classification/
│   ├── __init__.py             <- Package entry point
│   ├── config.py               <- Central path and environment configuration
│   ├── dataset.py              <- Dataset download, loading, and splitting helpers
│   ├── early_stopping.py       <- Early stopping utility
│   ├── evaluation.py           <- Metrics and confusion-matrix helpers
│   └── wanddb_helpers.py       <- Weights & Biases logging helpers
├── models/                     <- Saved model checkpoints
├── notebooks/                  <- Analysis and training notebooks
├── pyproject.toml              <- Package metadata and tooling configuration
└── wandb/                      <- Local W&B run artifacts
```

The source package is intentionally small. Most of the project logic is split between data
loading, training support, and evaluation helpers so the notebooks can stay focused on the
experimental flow rather than repeating boilerplate.

## More information

If you want the full context behind the design choices, results, and limitations, start with the
report in [Reports/DL_Project_Sophia_Tomer.pdf](Reports/DL_Project_Sophia_Tomer.pdf). It covers
the dataset setup, the single-label and multi-label comparisons, the per-class evaluation results,
the threshold sweep for the multi-label ViT, and the final discussion of limitations and future
work.

The main takeaways from the report are:

- ResNet50 is an excellent baseline for the single-label benchmark and reaches near-perfect test
  performance.
- The multi-label task benefits from a partially unfrozen ViT backbone, which adapts the pretrained
  features better than a fully frozen model.
- Simple augmentation and dropout did not outperform the best unfrozen ViT configuration in the
  experiments reported here.

That report is the best place to go if you need figures, tables, and the full discussion of the
experimental tradeoffs.



