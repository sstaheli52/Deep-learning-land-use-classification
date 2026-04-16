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

    # Split dataset into test/train
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    return train_df, test_df, class_names, num_classes

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
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    return train_df, test_df, class_names, num_classes