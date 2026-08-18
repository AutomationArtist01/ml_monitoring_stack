# MLflow tracking server + model registry (sqlite backend, local artifact store)
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
# --allowed-hosts: MLflow 3.x rejects unknown Host headers (DNS-rebinding guard); allow the
#   in-network name (mlflow:5000) and the host-mapped port (localhost:5001).
# --expose-prometheus: MLflow itself exports /metrics so Prometheus can scrape it too.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", "--serve-artifacts", \
     "--allowed-hosts", "mlflow:5000,localhost:5001,127.0.0.1:5001,localhost:5000,127.0.0.1:5000,mlflow", \
     "--cors-allowed-origins", "*", \
     "--expose-prometheus", "/mlflow/metrics"]
