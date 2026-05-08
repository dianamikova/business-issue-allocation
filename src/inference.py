import json

import joblib
import numpy as np

from src.config import DEFAULT_TARGET_COLUMN, DEFAULT_TEXT_COLUMN, LABEL_ENCODER_PATH, MODEL_PATH, TASK_CONFIG_PATH
from src.embeddings import encode_texts, load_embedding_model


class Predictor:
    def __init__(self) -> None:
        self.embedding_model = load_embedding_model()
        self.classifier = joblib.load(MODEL_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        self.task_config = self._load_task_config()

    def _load_task_config(self) -> dict:
        try:
            return json.loads(TASK_CONFIG_PATH.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            return joblib.load(TASK_CONFIG_PATH)

    def predict(self, text: str) -> dict:
        features = encode_texts(self.embedding_model, [text])
        prediction = self.classifier.predict(features)[0]
        label = self.label_encoder.inverse_transform([prediction])[0]
        scores = self._get_scores(features)[0]
        probabilities = self._scores_to_probabilities(scores)
        labels = self.label_encoder.inverse_transform(np.arange(len(probabilities)))
        ranked = sorted(
            zip(labels, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
        return {
            "text_column": self.task_config.get("text_column", DEFAULT_TEXT_COLUMN),
            "target_column": self.task_config.get("target_column", DEFAULT_TARGET_COLUMN),
            "predicted_label": label,
            "confidence": float(max(probabilities)),
            "top_predictions": [
                {"label": ranked_label, "confidence": float(confidence)}
                for ranked_label, confidence in ranked[:3]
            ],
        }

    def _get_scores(self, features):
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(features)
        if hasattr(self.classifier, "decision_function"):
            scores = self.classifier.decision_function(features)
            return np.atleast_2d(scores)
        predictions = self.classifier.predict(features)
        one_hot = np.zeros((len(predictions), len(self.label_encoder.classes_)))
        one_hot[np.arange(len(predictions)), predictions] = 1.0
        return one_hot

    def _scores_to_probabilities(self, scores):
        scores = np.asarray(scores, dtype=float)
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        total = np.sum(exp_scores)
        if total == 0:
            return np.ones_like(exp_scores) / len(exp_scores)
        return exp_scores / total
