from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
RESULTS_PATH = ARTIFACTS_DIR / "all_results.json"
TASK_CONFIG_PATH = ARTIFACTS_DIR / "task_config.json"

#EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_TEXT_COLUMN = "query"
DEFAULT_ID_COLUMN = "query_id"
DEFAULT_TARGET_COLUMN = "label_it"
