from zenml import pipeline

from src.steps.ingest import ingest_data
from src.steps.validate import validate_data
from src.steps.split import split_data
from src.steps.preprocess import preprocess_data
from src.steps.train import train_model
from src.steps.evaluate import evaluate_model


@pipeline
def customer_churn_pipeline():
    """Customer churn MLOps pipeline."""

    df = ingest_data()

    df = validate_data(df)

    train_df, test_df = split_data(df)

    train_processed, test_processed = preprocess_data(
        train_df,
        test_df,
    )

    model = train_model(train_processed)

    evaluate_model(
        model,
        test_processed,
    )
