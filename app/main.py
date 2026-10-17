import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field

from src.features import FEATURES

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("heart-api")

REQUESTS = Counter("api_requests_total", "HTTP requests", ["path", "method", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["path"])
PREDICTIONS = Counter("api_predictions_total", "Predictions by class", ["label"])

state = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    path = os.getenv("MODEL_PATH", "models/model.joblib")
    state["model"] = joblib.load(path)
    log.info("Model loaded from %s", path)
    yield


app = FastAPI(title="Heart Disease Risk API", lifespan=lifespan)


class Patient(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=1, le=4)
    trestbps: float = Field(..., gt=0)
    chol: float = Field(..., gt=0)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., gt=0)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0)
    slope: int = Field(..., ge=1, le=3)
    ca: Optional[float] = Field(None, ge=0, le=3)
    thal: Optional[float] = None


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    path = request.url.path
    REQUESTS.labels(path, request.method, response.status_code).inc()
    LATENCY.labels(path).observe(elapsed)
    log.info("%s %s -> %s (%.1f ms)", request.method, path,
             response.status_code, elapsed * 1000)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: Patient):
    row = pd.DataFrame([patient.model_dump()])[FEATURES].astype(float)
    p_disease = float(state["model"].predict_proba(row)[0, 1])
    pred = int(p_disease >= 0.5)
    PREDICTIONS.labels(str(pred)).inc()
    log.info("prediction=%d p_disease=%.3f input=%s", pred, p_disease, patient.model_dump())
    return {"prediction": pred,
            "label": "heart disease" if pred else "no heart disease",
            "confidence": round(p_disease if pred else 1 - p_disease, 4),
            "probability_disease": round(p_disease, 4)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
