from src.eda import run


def test_eda_writes_all_figures(tmp_path):
    figs = run(out=tmp_path)
    assert len(figs) == 6
    assert all(p.exists() and p.stat().st_size > 0 for p in figs)
