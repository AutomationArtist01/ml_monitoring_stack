import pandas as pd
import mlflow

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from zenml import step


@step
def evaluate_model(
    model,
    test_df: pd.DataFrame,
) -> None:
    """Evaluate the trained model on the test dataset."""

    print("Starting model evaluation...")

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    print(f"Test samples: {len(test_df)}")
    print(f"Test features: {X_test.shape[1]}")
    print(f"Target values: {y_test.unique()}")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    # Handle Yes/No target labels.
    if y_test.dtype == "object" or str(y_test.dtype) == "str":
        y_test_binary = (y_test == "Yes").astype(int)

        if set(predictions) <= {"Yes", "No"}:
            predictions_binary = (predictions == "Yes").astype(int)
        else:
            predictions_binary = predictions
    else:
        y_test_binary = y_test.astype(int)
        predictions_binary = predictions

    precision = precision_score(
        y_test_binary,
        predictions_binary,
        zero_division=0,
    )

    recall = recall_score(
        y_test_binary,
        predictions_binary,
        zero_division=0,
    )

    f1 = f1_score(
        y_test_binary,
        predictions_binary,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test_binary,
        probabilities,
    )

    cm = confusion_matrix(
        y_test_binary,
        predictions_binary,
    )

    print("\nTest Set Results")
    print("----------------")
    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall   : {recall:.4f}")
    print(f"Test F1 Score : {f1:.4f}")
    print(f"Test ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    mlflow.set_experiment(
        "customer_churn_evaluation"
    )

    with mlflow.start_run(
        run_name="test_set_evaluation"
    ):

        mlflow.log_metric(
            "test_accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "test_precision",
            precision,
        )

        mlflow.log_metric(
            "test_recall",
            recall,
        )

        mlflow.log_metric(
            "test_f1_score",
            f1,
        )

        mlflow.log_metric(
            "test_roc_auc",
            roc_auc,
        )

        mlflow.log_param(
            "test_samples",
            len(test_df),
        )

        mlflow.log_dict(
            {
                "labels": ["0", "1"],
                "matrix": cm.tolist(),
            },
            "confusion_matrix.json",
        )

    print("\nModel evaluation completed successfully.")
