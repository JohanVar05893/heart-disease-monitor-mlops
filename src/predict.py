"""Batch / single-record inference with the saved model.

Usage:
    python -m src.predict --json '{"age":63,"sex":1,...}'
    python -m src.predict --csv data/heart_clean.csv --out predictions.csv
"""
import argparse
import json
import sys

import joblib
import pandas as pd

from src.features import FEATURES

DEFAULT_MODEL = "models/model.joblib"


def predict_frame(df: pd.DataFrame, model_path: str = DEFAULT_MODEL) -> pd.DataFrame:
    """Return prediction, label and confidence for each row of `df`."""
    model = joblib.load(model_path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    proba = model.predict_proba(df[FEATURES].astype(float))[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pd.DataFrame({
        "prediction": pred,
        "label": ["heart disease" if p else "no heart disease" for p in pred],
        "confidence": [round(float(p if y else 1 - p), 4) for p, y in zip(proba, pred)],
        "probability_disease": proba.round(4)})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", help="one patient record as a JSON string")
    src.add_argument("--csv", help="CSV file with the feature columns")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", help="write CSV results here (csv mode)")
    args = ap.parse_args(argv)

    df = pd.DataFrame([json.loads(args.json)]) if args.json else pd.read_csv(args.csv)
    res = predict_frame(df, args.model)
    if args.out:
        res.to_csv(args.out, index=False)
        print(f"wrote {len(res)} predictions to {args.out}")
    else:
        print(res.to_json(orient="records", indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
