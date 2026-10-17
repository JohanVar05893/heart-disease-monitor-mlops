"""Train, tune, cross-validate and track candidate models with MLflow."""
import json
from pathlib import Path

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,  # noqa: E402
                             accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,  # noqa: E402
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline  # noqa: E402

from src.data import CLEAN_PATH  # noqa: E402
from src.features import FEATURES, build_preprocessor  # noqa: E402

SEED = 42
SCORING = ["accuracy", "precision", "recall", "roc_auc"]
CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000),
        {"clf__C": [0.01, 0.1, 1, 10]}),
    "random_forest": (
        RandomForestClassifier(random_state=SEED),
        {"clf__n_estimators": [100, 300],
         "clf__max_depth": [3, 5, None],
         "clf__min_samples_leaf": [1, 3]}),
}


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", estimator)])


def load_split(path=CLEAN_PATH):
    df = pd.read_csv(path)
    return train_test_split(df[FEATURES], df["target"], test_size=0.2,
                            stratify=df["target"], random_state=SEED)


def main(model_dir: str = "models") -> None:
    X_tr, X_te, y_tr, y_te = load_split()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("heart-disease")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best = {"cv_roc_auc": -1}

    for name, (est, grid) in CANDIDATES.items():
        with mlflow.start_run(run_name=name):
            gs = GridSearchCV(make_pipeline(est), grid, cv=cv,
                              scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
            model = gs.best_estimator_
            mlflow.log_param("model_type", name)
            mlflow.log_params(gs.best_params_)

            cvres = cross_validate(model, X_tr, y_tr, cv=cv, scoring=SCORING)
            metrics = {}
            for m in SCORING:
                metrics[f"cv_{m}"] = cvres[f"test_{m}"].mean()
                metrics[f"cv_{m}_std"] = cvres[f"test_{m}"].std()

            proba = model.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            metrics.update({
                "test_accuracy": accuracy_score(y_te, pred),
                "test_precision": precision_score(y_te, pred),
                "test_recall": recall_score(y_te, pred),
                "test_roc_auc": roc_auc_score(y_te, proba)})
            mlflow.log_metrics(metrics)

            fig, ax = plt.subplots()
            RocCurveDisplay.from_predictions(y_te, proba, ax=ax, name=name)
            mlflow.log_figure(fig, "roc_curve.png")
            plt.close(fig)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax)
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

            mlflow.sklearn.log_model(model, "model")
            print(name, {k: round(v, 3) for k, v in metrics.items()})
            if metrics["cv_roc_auc"] > best["cv_roc_auc"]:
                best = {"name": name, "model": model, **metrics}

    out = Path(model_dir)
    out.mkdir(exist_ok=True)
    joblib.dump(best.pop("model"), out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(best, indent=2))
    print("Selected:", best["name"])


if __name__ == "__main__":
    main()
