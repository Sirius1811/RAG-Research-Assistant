from langchain_community.embeddings import HuggingFaceEmbeddings
from config.config import EMBEDDING_MODEL


def get_embedding_model():
    """
    Initialize and return the HuggingFace sentence-transformer embedding model.
    Cached via lru_cache at the call site to avoid reloading on each invocation.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embeddings: {str(e)}")
