import pandas as pd

from src.data import COLUMNS, clean


def _write_raw(tmp_path, rows):
    path = tmp_path / "raw.data"
    path.write_text("\n".join(",".join(map(str, r)) for r in rows))
    return path


def test_clean_handles_missing_marker_binarises_target_and_dedupes(tmp_path):
    base = [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]
    rows = [base + [0],
            base + [0],                                # exact duplicate
            [50, 0, 2, 120, 200, 0, 0, 160, 0, 1.0, 1, "?", 3, 3],
            [41, 1, 3, 130, 190, 0, 0, 170, 1, 0.5, 2, 1, "?", 1]]
    df = clean(_write_raw(tmp_path, rows))
    assert list(df.columns) == COLUMNS
    assert len(df) == 3
    assert df["target"].tolist() == [0, 1, 1]
    assert df["ca"].isna().sum() == 1 and df["thal"].isna().sum() == 1
    assert pd.api.types.is_numeric_dtype(df["ca"])
