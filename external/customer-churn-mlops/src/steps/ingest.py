import pandas as pd
from zenml import step


@step
def ingest_data() -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset."""

    file_path = "data/ML DATASET_Telco-Customer-Churn.csv"

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df
