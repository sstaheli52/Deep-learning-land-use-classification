from pathlib import Path

from dotenv import load_dotenv
import subprocess
import shutil
import zipfile
import tempfile
import pandas as pd
from sklearn.model_selection import train_test_split

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

IMAGES_DIR = RAW_DATA_DIR / "Images"
LABELS_FILE = RAW_DATA_DIR / "LandUse_Multilabeled.txt"

WANDB_DIR = PROJ_ROOT / "wandb"

# Download and extract the dataset if not already present
if not IMAGES_DIR.exists() or not LABELS_FILE.exists():   
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "ucmdata"
        # Clone repo
        subprocess.run(["git", "clone", "--depth", "1", "https://git.wur.nl/lobry001/ucmdata.git", str(repo)], check=True)
        # Copy labels file
        shutil.copy2(repo / "LandUse_Multilabeled.txt", LABELS_FILE)
        # Unzip images and move to target directory
        with zipfile.ZipFile(repo / "UCMerced_LandUse.zip") as z:
            z.extractall(tmp)
        shutil.move(str(Path(tmp) / "UCMerced_LandUse" / "Images"), IMAGES_DIR)
        shutil.rmtree(repo)

# -- Data preparation --
# Get the multilabels data as a DataFrame
df = pd.read_csv(LABELS_FILE, sep=r"\s+", header=0)

# get path for images from 'IMAGE\LABEL' column
parts = df["IMAGE\\LABEL"].str.extract(r"^([^\d]+)(\d+)$")
df["folder"] = parts[0]
df["number"] = parts[1]

df["path"] = (
    str(IMAGES_DIR) + "/" + df["folder"] + "/" + df["folder"] + df["number"] + ".tif"
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