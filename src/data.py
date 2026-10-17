"""Download the UCI Heart Disease (Cleveland) data and produce a clean CSV."""
from pathlib import Path

import pandas as pd
import requests

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
RAW_PATH = Path("data/raw/processed.cleveland.data")
CLEAN_PATH = Path("data/heart_clean.csv")


def download(dest: Path = RAW_PATH) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def clean(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """'?' -> NaN, binarise target (0 = healthy, 1-4 = disease), drop duplicates.

    Missing values (ca, thal) are kept as NaN here and imputed inside the
    model pipeline so imputation statistics are learned on training data only.
    """
    df = pd.read_csv(raw_path, header=None, names=COLUMNS, na_values="?")
    df["target"] = (df["target"] > 0).astype(int)
    return df.drop_duplicates().reset_index(drop=True)


if __name__ == "__main__":
    clean(download()).to_csv(CLEAN_PATH, index=False)
    print(f"Saved {CLEAN_PATH}")
