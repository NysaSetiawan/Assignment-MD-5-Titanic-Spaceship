from pathlib import Path
import joblib

ARTIFACT_DIR = Path("artifacts")
PIPELINE_PATH = ARTIFACT_DIR / "pipeline.pkl"


def save_pipeline(pipeline) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)


def load_pipeline():
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Pipeline not found: {PIPELINE_PATH}\n"
            "Run training pipeline first."
        )
    return joblib.load(PIPELINE_PATH)