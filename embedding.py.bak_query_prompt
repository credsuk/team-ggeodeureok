import torch
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_path: str):
    """로컬 임베딩 모델을 로딩한다."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return SentenceTransformer(
        model_path,
        device=device,
        local_files_only=True,
    )


def encode_query(model, query: str):
    """검색어를 임베딩 벡터로 변환한다."""
    return model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )


# 임베딩 기본 코드 -----------------------------------------------------------------------------------------------