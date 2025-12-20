from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Dict

from sentence_transformers import SentenceTransformer

from .config import Settings, EmbeddingModelConfig
from .kg_client import KGClient


@lru_cache(maxsize=4)
def _load_model(name: str) -> SentenceTransformer:
    """Load and cache embedding model."""
    return SentenceTransformer(name)


class EmbeddingService:
    def __init__(self, settings: Settings, model_key: str = "model_1"):
        self.settings = settings
        self.model_key = model_key
        
        # Get the model configuration
        models = settings.get_embedding_models()
        if model_key not in models:
            raise ValueError(f"Model key '{model_key}' not found. Available: {list(models.keys())}")
        
        self.model_config = models[model_key]
        self.model = _load_model(self.model_config.model_id)

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed texts using the selected model."""
        vectors = self.model.encode(list(texts), convert_to_numpy=True)
        return [vec.tolist() for vec in vectors]

    def semantic_search(
        self, client: KGClient, query: str, top_k: int = 10
    ) -> List[dict]:
        """Perform semantic search using vector queries."""
        try:
            vector = self.embed([query])[0]
            return client.vector_query(
                vector=vector,
                top_k=top_k,
                index_name=self.model_config.vector_index,
            )
        except Exception as e:
            # Return empty list if vector search fails
            print(f"Warning: Vector search failed for query '{query}': {e}")
            return []