# Deep-learning-land-use-classification

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

WUR MGI project to classify land use images.

## Get started

1. Create a new python evironment.
2. Run 'pip install -e .'
3. To use wandb:
    1. Activate your python environment
    2. Follow the steps in the [wandb documentation](https://docs.wandb.ai/models/quickstart#command-line).

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.│
├── models             <- Trained and serialized models, model predictions, or model summaries
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         deep_learning_land_use_classification and configuration for tools like black│││
└── deep_learning_land_use_classification   <- Source code for use in this project.
    ├── __init__.py             <- Makes deep_learning_land_use_classification a Python module
    ├── config.py               <- Store useful variables and configuration
    ├── dataset.py              <- Scripts to download or generate data
    ├── early_stopping.py       <- Script for early stopping class
    ├── dataset.py              <- Script for 
```



