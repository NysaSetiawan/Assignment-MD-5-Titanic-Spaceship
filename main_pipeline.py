from config import ACCURACY_THRESHOLD
from src.data import ingest_data, load_frame, split_features_target, split_train_test
from src.pipelines import build_pipeline
from src.models import train_pipeline, evaluate



def main():
    print("=" * 50)
    print("Spaceship Titanic")
    print("=" * 50)

    print("\nStep 1: Data Ingestion")
    ingest_data()

    print("\nStep 2: Load and Split")
    df = load_frame()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print("\nStep 3: Build Pipeline")
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    pipeline = build_pipeline(num_features, cat_features)

    print("\nStep 4: Train Pipeline")
    run_id, model = train_pipeline(pipeline, X_train, y_train)

    print("\nStep 5: Evaluation")
    accuracy, precision, recall = evaluate(model, X_test, y_test, run_id)


    print("\n" + "=" * 50)
    if accuracy >= ACCURACY_THRESHOLD:
        print(f"Model Approved (accuracy={accuracy:.3f})")
    else:
        print(f"Model Rejected (accuracy={accuracy:.3f} < {ACCURACY_THRESHOLD})")


if __name__ == "__main__":
    main()