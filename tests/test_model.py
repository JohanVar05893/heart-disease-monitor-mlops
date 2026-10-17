import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.train import make_pipeline


def test_pipeline_trains_and_beats_chance(split):
    X_tr, X_te, y_tr, y_te = split
    model = make_pipeline(RandomForestClassifier(n_estimators=50, random_state=0))
    proba = model.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))
    assert roc_auc_score(y_te, proba) > 0.7
