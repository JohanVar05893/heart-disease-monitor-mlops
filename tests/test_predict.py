import json

import pandas as pd
import pytest

from src.features import FEATURES
from src.predict import main, predict_frame

SAMPLE = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
          "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3,
          "slope": 3, "ca": 0, "thal": 6}


def test_predict_frame_columns_and_ranges(tiny_model):
    res = predict_frame(pd.DataFrame([SAMPLE]), str(tiny_model))
    assert set(res.columns) == {"prediction", "label", "confidence", "probability_disease"}
    assert 0.5 <= res.loc[0, "confidence"] <= 1


def test_predict_frame_rejects_missing_columns(tiny_model):
    with pytest.raises(ValueError):
        predict_frame(pd.DataFrame([{"age": 50}]), str(tiny_model))


def test_cli_json_and_csv(tiny_model, tmp_path, capsys):
    assert main(["--json", json.dumps(SAMPLE), "--model", str(tiny_model)]) == 0
    assert json.loads(capsys.readouterr().out)[0]["prediction"] in (0, 1)

    src = tmp_path / "in.csv"
    pd.DataFrame([SAMPLE] * 3)[FEATURES].to_csv(src, index=False)
    out = tmp_path / "out.csv"
    main(["--csv", str(src), "--model", str(tiny_model), "--out", str(out)])
    assert len(pd.read_csv(out)) == 3
