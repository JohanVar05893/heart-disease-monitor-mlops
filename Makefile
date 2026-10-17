.PHONY: setup data eda train mlflow test lint predict docker-build docker-run k8s-up k8s-down monitoring-up monitoring-down

PY ?= python
SAMPLE := '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'

setup:
	python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt

data:
	$(PY) -m src.data

eda:
	$(PY) -m src.eda

train:
	$(PY) -m src.train

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

lint:
	flake8 src app tests docs

test:
	pytest -v

predict:
	$(PY) -m src.predict --json $(SAMPLE)

docker-build:
	docker build -t heart-api:1.0 .

docker-run:
	docker run -d --rm -p 8000:8000 --name heart-api heart-api:1.0
	@sleep 5 && curl -s localhost:8000/health && echo
	curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d $(SAMPLE) && echo

k8s-up:
	kubectl apply -f k8s/ && kubectl rollout status deployment/heart-api && kubectl get pods,svc

k8s-down:
	kubectl delete -f k8s/

monitoring-up:
	cd monitoring && docker compose up --build -d

monitoring-down:
	cd monitoring && docker compose down
