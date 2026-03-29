from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.features.pipeline_preprocessor import build_preprocessor

RANDOM_STATE = 42


def build_pipeline(num_features: list, cat_features: list) -> Pipeline:
    preprocessor = build_preprocessor(num_features, cat_features)

    model = LogisticRegression(
        solver="liblinear",
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model),
    ])

    return pipeline