import pandas as pd

from src.data import CLEAN_PATH, COLUMNS
from src.features import FEATURES, build_preprocessor


def test_clean_schema_and_target():
    df = pd.read_csv(CLEAN_PATH)
    assert list(df.columns) == COLUMNS
    assert set(df["target"].unique()) <= {0, 1}
    assert not df.duplicated().any()


def test_preprocessor_outputs_no_nans_even_with_missing_inputs():
    df = pd.read_csv(CLEAN_PATH)[FEATURES]
    out = build_preprocessor().fit_transform(df)
    arr = out.toarray() if hasattr(out, "toarray") else out
    assert not pd.isna(arr).any()
