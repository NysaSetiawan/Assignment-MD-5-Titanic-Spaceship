import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score


def evaluate(model, X_val, y_val, run_id):

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, average="binary")
    rec = recall_score(y_val, preds, average="binary")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_precision", prec)
        mlflow.log_metric("val_recall", rec)

    print(
        f"Evaluation | Accuracy={acc:.3f} | Precision={prec:.3f} | Recall={rec:.3f}"
    )

    return acc, prec, rec

