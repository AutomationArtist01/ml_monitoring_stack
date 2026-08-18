import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step


@step
def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the cleaned dataset into training and testing data."""

    print("Starting train/test split...")

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    # 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Add target back temporarily so the next ZenML step
    # receives complete datasets.
    train_df = X_train.copy()
    train_df["Churn"] = y_train.values

    test_df = X_test.copy()
    test_df["Churn"] = y_test.values

    print(f"Training dataset shape: {train_df.shape}")
    print(f"Testing dataset shape: {test_df.shape}")

    print(
        f"Training churn rate: "
        f"{train_df['Churn'].mean():.4f}"
    )

    print(
        f"Testing churn rate: "
        f"{test_df['Churn'].mean():.4f}"
    )

    print("Train/test split completed successfully.")

    return train_df, test_df
