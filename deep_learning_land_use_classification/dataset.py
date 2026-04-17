# Internal imports
import deep_learning_land_use_classification.config as config
# External imports
import subprocess
import shutil
import zipfile
import tempfile
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Multilabel stratified splitting using iterative-stratification
# Author: Trent J. Bradberry (2018)
# License: BSD 3-Clause
# Based on: Sechidis et al. (2011)
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def download_data():
    # Download and extract the dataset if not already present
    if not config.IMAGES_DIR.exists() or not config.LABELS_FILE.exists():   
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "ucmdata"
            # Clone repo
            subprocess.run(["git", "clone", "--depth", "1", "https://git.wur.nl/lobry001/ucmdata.git", str(repo)], check=True)
            # Copy labels file
            shutil.copy2(repo / "LandUse_Multilabeled.txt", config.LABELS_FILE)
            # Unzip images and move to target directory
            with zipfile.ZipFile(repo / "UCMerced_LandUse.zip") as z:
                z.extractall(tmp)
            shutil.move(str(Path(tmp) / "UCMerced_LandUse" / "Images"), config.IMAGES_DIR)
            shutil.rmtree(repo)
    
def get_multi_label_data():
    # Download data if not already present
    download_data()
    
    # Get the multilabels data as a DataFrame
    df = pd.read_csv(config.LABELS_FILE, sep=r"\s+", header=0)

    # get path for images from 'IMAGE\LABEL' column
    parts = df["IMAGE\\LABEL"].str.extract(r"^([^\d]+)(\d+)$")
    df["folder"] = parts[0]
    df["number"] = parts[1]

    df["path"] = (
        str(config.IMAGES_DIR) + "/" + df["folder"] + "/" + df["folder"] + df["number"] + ".tif"
    )

    # Get class columns
    class_names = df.columns[1:-3]
    num_classes = len(class_names)

    # Split dataset into test and train
    X = df.index.values
    y = df[class_names].values

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(msss.split(X, y))
    train_val_int = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # split train into validation and train
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=43)
    train_idx, val_idx = next(msss_val.split(train_val_int.index.values, train_val_int[class_names].values))
    val_df = train_val_int.iloc[val_idx]
    train_df = train_val_int.iloc[train_idx] 
 
    
    return train_df, test_df, val_df, class_names, num_classes

def get_single_label_data():
    # Download data if not already present
    download_data()
    
    # Get class names from subdirectories
    class_names = sorted([d.name for d in config.IMAGES_DIR.iterdir() if d.is_dir()])
    num_classes = len(class_names)
    
    # Create DataFrame with image paths and labels
    data = []
    for class_name in class_names:
        class_dir = config.IMAGES_DIR / class_name
        for image_path in class_dir.glob("*.tif"):
            data.append({"path": str(image_path), "label": class_name})
    
    df = pd.DataFrame(data)
    
    # Split dataset into train/test
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    # split train into validation and train
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.2,
        random_state=43,
        stratify=train_val_df["label"]
    )
    
    return train_df, test_df, val_df, class_names, num_classes