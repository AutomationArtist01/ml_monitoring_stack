import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from zenml import step


@step
def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess train and test data without data leakage."""

    print("Starting preprocessing...")

    # Separate targets
    y_train = train_df["Churn"]
    y_test = test_df["Churn"]

    X_train = train_df.drop(columns=["Churn", "customerID"])
    X_test = test_df.drop(columns=["Churn", "customerID"])

    numerical_features = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ]
    )

    # IMPORTANT:
    # Fit ONLY on training data.
    X_train_processed = preprocessor.fit_transform(X_train)

    # Only transform test data.
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index,
    )

    X_train_processed["Churn"] = y_train.values
    X_test_processed["Churn"] = y_test.values

    print(
        f"Training processed shape: "
        f"{X_train_processed.shape}"
    )

    print(
        f"Testing processed shape: "
        f"{X_test_processed.shape}"
    )

    print(
        f"Processed features: "
        f"{len(feature_names)}"
    )

    print("Preprocessing completed successfully.")

    return X_train_processed, X_test_processed
