import json
import pathlib

import joblib

from src import train

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "heart_clean.csv"
_real_load_split = train.load_split


def test_main_trains_both_models_and_saves_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep mlflow.db / mlruns out of the repo
    monkeypatch.setattr(train, "CANDIDATES", {
        "logistic_regression": (train.LogisticRegression(max_iter=500), {"clf__C": [1]}),
        "random_forest": (train.RandomForestClassifier(random_state=0, n_estimators=10),
                          {"clf__max_depth": [3]}),
    })
    monkeypatch.setattr(train, "load_split", lambda: _real_load_split(DATA))
    train.main(model_dir=str(tmp_path / "models"))
    metrics = json.loads((tmp_path / "models" / "metrics.json").read_text())
    assert metrics["name"] in {"logistic_regression", "random_forest"}
    assert 0.5 < metrics["cv_roc_auc"] <= 1
    assert joblib.load(tmp_path / "models" / "model.joblib") is not None
