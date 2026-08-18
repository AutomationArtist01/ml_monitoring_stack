import os
import time

import mlflow
import pandas as pd
from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Customer churn prediction using MLflow "
        "and Gradient Boosting"
    ),
    version="1.0.0",
)


# ============================================================
# Prometheus monitoring
# ============================================================

Instrumentator().instrument(app).expose(app)

prediction_requests = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)

prediction_errors = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
)

prediction_latency = Histogram(
    "prediction_request_latency_seconds",
    "Prediction request latency in seconds",
)


# ============================================================
# Input schema
# ============================================================

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# ============================================================
# Feature definitions
# ============================================================

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


# ============================================================
# Preprocessing
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_features,
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
    ]
)


# ============================================================
# Dataset path
# ============================================================

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "ML DATASET_Telco-Customer-Churn.csv",
)

print("Initializing API preprocessing...")


# ============================================================
# Fit preprocessing using dataset
# ============================================================

try:
    dataset = pd.read_csv(DATASET_PATH)

    dataset["TotalCharges"] = pd.to_numeric(
        dataset["TotalCharges"],
        errors="coerce",
    )

    dataset["TotalCharges"] = dataset[
        "TotalCharges"
    ].fillna(0)

    X = dataset.drop(
        columns=["customerID", "Churn"],
        errors="ignore",
    )

    preprocessor.fit(X)

    feature_count = len(
        preprocessor.get_feature_names_out()
    )

    print("API preprocessing initialized.")
    print(
        f"Number of processed features: {feature_count}"
    )

except Exception as e:
    print("ERROR: Could not initialize preprocessing.")
    print("Error:", e)

    raise RuntimeError(
        "API preprocessing initialization failed."
    ) from e


# ============================================================
# Load MLflow model
# ============================================================

if os.path.exists("/app/model"):
    MODEL_PATH = "/app/model"

elif os.path.exists(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model",
    )
):
    MODEL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model",
    )

else:
    raise RuntimeError(
        "Production model not found. "
        "Expected /app/model or local model directory."
    )


print("Loading production model...")

try:
    model = mlflow.sklearn.load_model(
        MODEL_PATH
    )

    print(
        "Production Gradient Boosting model "
        "loaded successfully."
    )

except Exception as e:
    print(
        "ERROR: Could not load production model."
    )
    print("Error:", e)

    raise RuntimeError(
        "Production model loading failed."
    ) from e


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "CustomerChurnGradientBoosting",
        "features": feature_count,
    }


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
def predict(customer: CustomerData):

    start_time = time.perf_counter()

    prediction_requests.inc()

    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame(
            [customer.model_dump()]
        )

        # Apply same preprocessing
        processed_data = preprocessor.transform(
            input_data
        )

        # Make prediction
        prediction_value = int(
            model.predict(processed_data)[0]
        )

        # Get probability
        probabilities = model.predict_proba(
            processed_data
        )[0]

        churn_probability = float(
            probabilities[1]
        )

        # Convert prediction to label
        if prediction_value == 1:
            churn_result = "Yes"
        else:
            churn_result = "No"

        return {
            "prediction": prediction_value,
            "churn": churn_result,
            "churn_probability": round(
                churn_probability,
                4,
            ),
        }

    except Exception:
        prediction_errors.inc()
        raise

    finally:
        elapsed_time = (
            time.perf_counter() - start_time
        )

        prediction_latency.observe(
            elapsed_time
        )
