from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    return model.encode(texts, convert_to_numpy=True)
