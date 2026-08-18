import pandas as pd
import optuna
import mlflow
import mlflow.sklearn

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from zenml import step


@step
def train_model(train_df: pd.DataFrame):
    """Train Gradient Boosting model using Optuna."""

    print("Starting Optuna hyperparameter tuning...")

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]

    mlflow.set_experiment("customer_churn_optuna")

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, 300
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 2, 8
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 20
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 10
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.6, 1.0
            ),
        }

        model = GradientBoostingClassifier(
            **params,
            random_state=42,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_train)
        probabilities = model.predict_proba(X_train)[:, 1]

        accuracy = accuracy_score(
            y_train,
            predictions,
        )

        f1 = f1_score(
            y_train,
            predictions,
        )

        roc_auc = roc_auc_score(
            y_train,
            probabilities,
        )

        with mlflow.start_run(
            run_name=f"optuna_trial_{trial.number}"
        ):

            mlflow.log_params(params)

            mlflow.log_metric(
                "accuracy",
                accuracy,
            )

            mlflow.log_metric(
                "f1_score",
                f1,
            )

            mlflow.log_metric(
                "roc_auc",
                roc_auc,
            )

        return roc_auc

    study = optuna.create_study(
        direction="maximize",
        study_name="customer_churn_gradient_boosting",
    )

    study.optimize(
        objective,
        n_trials=35,
    )

    print("\nOptuna tuning completed.")

    print(
        f"Number of trials: {len(study.trials)}"
    )

    print(
        f"Best ROC-AUC: {study.best_value:.4f}"
    )

    print("Best parameters:")

    for parameter, value in study.best_params.items():
        print(f"  {parameter}: {value}")

    # Train final model using best parameters
    best_model = GradientBoostingClassifier(
        **study.best_params,
        random_state=42,
    )

    best_model.fit(
        X_train,
        y_train,
    )

    predictions = best_model.predict(X_train)
    probabilities = best_model.predict_proba(X_train)[:, 1]

    accuracy = accuracy_score(
        y_train,
        predictions,
    )

    f1 = f1_score(
        y_train,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_train,
        probabilities,
    )

    # Log final best model to MLflow
    with mlflow.start_run(
        run_name="best_gradient_boosting_model"
    ):

        mlflow.log_params(
            study.best_params
        )

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "f1_score",
            f1,
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc,
        )

        mlflow.sklearn.log_model(
            best_model,
            name="gradient_boosting_model",
        )

    print("\nFinal model trained successfully.")

    print(f"Training Accuracy: {accuracy:.4f}")
    print(f"Training F1 Score: {f1:.4f}")
    print(f"Training ROC-AUC: {roc_auc:.4f}")

    return best_model
