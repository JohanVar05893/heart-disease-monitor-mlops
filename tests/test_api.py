import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLE = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
          "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3,
          "slope": 3, "ca": 0, "thal": 6}


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1) and 0.5 <= body["confidence"] <= 1


def test_predict_rejects_invalid_input(client):
    assert client.post("/predict", json={**SAMPLE, "age": -5}).status_code == 422


def test_metrics_endpoint(client):
    assert b"api_requests_total" in client.get("/metrics").content
