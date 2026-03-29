from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_ING_DIR = BASE_DIR / "data" / "ingested"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

ARTIFACT_PIPELINE = ARTIFACTS_DIR / "pipeline.pkl"
ARTIFACT_MODEL = ARTIFACTS_DIR / "model.pkl"

TARGET_COL = "Transported"

NUM_FEATURES = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck"
]

CAT_FEATURES = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP"
]

DROP_COLS = [
    "PassengerId",
    "Cabin",
    "Name"
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
LOGREG_SOLVER = "liblinear"
LOGREG_MAX_ITER = 1000

MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR.parent / 'mlflow.db'}"
MLFLOW_EXP_PIPELINE = "Spaceship Titanic Pipeline"

ACCURACY_THRESHOLD = 0.77