# Customer Churn MLOps

End-to-end Customer Churn Prediction MLOps project.

## Features

- Customer churn prediction using Gradient Boosting
- Optuna hyperparameter tuning
- MLflow experiment tracking and model logging
- ZenML pipeline
- FastAPI prediction API
- Docker containerization
- GitHub Actions CI/CD
- Prometheus metrics
- Grafana monitoring dashboard
- Deployment on Render

## MLOps Pipeline

Data → Training → Optuna Tuning → MLflow → Model → FastAPI → Docker → Render → Prometheus → Grafana

## API

Live API:

https://customer-churn-api-hzsz.onrender.com/

API documentation:

https://customer-churn-api-hzsz.onrender.com/docs

Metrics:

https://customer-churn-api-hzsz.onrender.com/metrics

## Monitoring

The project monitors:

- Prediction request count
- Prediction errors
- Prediction latency
- API availability
- HTTP request metrics

## CI/CD

GitHub Actions automatically runs:

- Linting and tests
- Docker image build

## Model

The project uses a Gradient Boosting Classifier with Optuna-based hyperparameter optimization and MLflow experiment tracking.
