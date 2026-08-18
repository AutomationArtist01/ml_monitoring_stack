import pandas as pd
from zenml import step


@step
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the Telco Customer Churn dataset."""

    print("Starting data validation...")

    # 1. Required columns
    required_columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
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
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Schema check: PASSED")

    # 2. Check duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")

    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows.")

    # 3. Handle blank TotalCharges values
    blank_total_charges = (
        df["TotalCharges"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"Blank TotalCharges values: {blank_total_charges}")

    # Convert blanks to missing values
    df["TotalCharges"] = (
        df["TotalCharges"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
    )

    # Convert to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Zero-tenure customers have zero total charges
    zero_tenure_mask = (
        df["tenure"].eq(0) &
        df["TotalCharges"].isna()
    )

    df.loc[zero_tenure_mask, "TotalCharges"] = 0.0

    # 4. Check remaining missing values
    missing_values = df.isna().sum().sum()

    print(f"Remaining missing values: {missing_values}")

    if missing_values > 0:
        raise ValueError(
            "Dataset still contains missing values after cleaning."
        )

    # 5. Validate target
    valid_churn_values = {"Yes", "No"}
    actual_churn_values = set(df["Churn"].unique())

    if not actual_churn_values.issubset(valid_churn_values):
        raise ValueError(
            f"Unexpected Churn values: {actual_churn_values}"
        )

    print("Target validation: PASSED")

    # 6. Final validation
    print("Data validation completed successfully.")
    print(f"Final dataset shape: {df.shape}")

    return df
