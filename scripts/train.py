import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ARTIFACTS_DIR,
    DEFAULT_ID_COLUMN,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TEXT_COLUMN,
    LABEL_ENCODER_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
    RESULTS_PATH,
    TASK_CONFIG_PATH,
)
from src.data_utils import load_csv, save_json, validate_dataframe
from src.embeddings import encode_texts, load_embedding_model
from src.model_utils import build_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", default="logreg", choices=["logreg", "svm", "random_forest"])
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--id-column", default=DEFAULT_ID_COLUMN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_csv(RAW_DATA_PATH)
    validate_dataframe(df, args.text_column, args.target_column, args.id_column)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[args.target_column],
    )

    embedding_model = load_embedding_model()
    x_train = encode_texts(embedding_model, train_df[args.text_column].tolist())
    x_test = encode_texts(embedding_model, test_df[args.text_column].tolist())

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[args.target_column])
    y_test = label_encoder.transform(test_df[args.target_column])

    classifier = build_classifier(args.classifier)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    metrics = {
        "classifier": args.classifier,
        "target_column": args.target_column,
        "text_column": args.text_column,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "label_distribution": dict(Counter(df[args.target_column])),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    joblib.dump(
        {
            "text_column": args.text_column,
            "target_column": args.target_column,
            "id_column": args.id_column,
        },
        TASK_CONFIG_PATH,
    )
    save_json(METRICS_PATH, metrics)
    if RESULTS_PATH.exists():
        all_results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    else:
        all_results = {"runs": []}
    all_results["runs"] = [run for run in all_results.get("runs", []) if run.get("classifier") != args.classifier]
    all_results["runs"].append(metrics)
    save_json(RESULTS_PATH, all_results)

    print(f"Saved classifier to {MODEL_PATH}")
    print(f"Saved label encoder to {LABEL_ENCODER_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(metrics)


if __name__ == "__main__":
    main()
