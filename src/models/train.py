import os
import mlflow
import mlflow.sklearn
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from utils.io import save_pipeline

RANDOM_STATE = 42


def train_pipeline(pipeline, X_train, y_train):

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Streamlit-Pipeline")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):

        params = {
            "model__C": trial.suggest_float("C", 0.001, 100, log=True),
            "model__l1_ratio": trial.suggest_float("l1_ratio", 0, 1),
            "model__solver": "saga",
            "model__max_iter": trial.suggest_int("max_iter", 2000, 5000),
            "model__random_state": RANDOM_STATE
        }

        pipeline.set_params(**params)

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )

        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )

    study.optimize(objective, n_trials=30)

    best_params = {
        "model__C": study.best_params["C"],
        "model__l1_ratio": study.best_params["l1_ratio"],
        "model__max_iter": study.best_params["max_iter"],
        "model__solver": "saga",
        "model__random_state": RANDOM_STATE
    }

    os.makedirs("artifacts", exist_ok=True)

    with mlflow.start_run() as run:

        pipeline.set_params(**best_params)

        pipeline.fit(X_train, y_train)

        cv_accuracy = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy"
        ).mean()

        mlflow.log_params(best_params)
        mlflow.log_metric("cv_accuracy", cv_accuracy)

        mlflow.sklearn.log_model(pipeline, "model")

        save_pipeline(pipeline)

        print(f"Best CV Accuracy: {cv_accuracy:.3f}")
        print(f"Run ID: {run.info.run_id}")

        return run.info.run_id, pipeline