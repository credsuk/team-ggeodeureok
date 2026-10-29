import torch
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_path: str):
    """권준 ZIP과 동일한 설정으로 로컬 임베딩 모델을 로딩한다."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(
        model_path,
        device=device,
        local_files_only=True,
    )
    model.max_seq_length = 2048
    return model


def encode_query(model, query: str):
    """권준 ZIP의 QueryEncoder와 동일한 방식으로 검색어를 임베딩한다."""
    kwargs = {
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }

    if "query" in getattr(model, "prompts", {}):
        kwargs["prompt_name"] = "query"

    return model.encode(query, **kwargs)
