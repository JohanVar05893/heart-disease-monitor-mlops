import os

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from src.train import load_split, make_pipeline


@pytest.fixture(scope="session")
def split():
    return load_split()


@pytest.fixture(scope="session", autouse=True)
def tiny_model(tmp_path_factory, split):
    """Train a quick model so API tests don't depend on a committed artifact."""
    X_tr, _, y_tr, _ = split
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    joblib.dump(make_pipeline(LogisticRegression(max_iter=500)).fit(X_tr, y_tr), path)
    os.environ["MODEL_PATH"] = str(path)
    return path
