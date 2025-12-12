from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

from sentence_transformers import SentenceTransformer

from .config import Settings
from .kg_client import KGClient


@lru_cache(maxsize=2)
def _load_model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = _load_model(settings.embed_model)

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        vectors = self.model.encode(list(texts), convert_to_numpy=True)
        return [vec.tolist() for vec in vectors]

    def semantic_search(
        self, client: KGClient, query: str, top_k: int = 10
    ) -> List[dict]:
        vector = self.embed([query])[0]
        return client.vector_query(vector=vector, top_k=top_k)
