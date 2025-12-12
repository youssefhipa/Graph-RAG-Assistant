from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from .config import get_settings
from .embedding import EmbeddingService
from .entities import EntityExtractor, EntityResult
from .intent import IntentClassifier
from .kg_client import KGClient
from .llm import LLMRegistry, run_llm
from .queries import build_query


@dataclass
class RetrievalResult:
    intent: str
    entities: EntityResult
    cypher: Optional[str]
    params: Dict[str, object]
    baseline_rows: List[Dict[str, object]]
    embed_rows: List[Dict[str, object]]
    answer: Optional[str] = None


class Pipeline:
    def __init__(self):
        self.settings = get_settings()
        self.intent = IntentClassifier()
        self.entities = EntityExtractor()
        self.embeddings = EmbeddingService(self.settings)
        self.llm_registry = LLMRegistry(self.settings)

    def run(
        self,
        question: str,
        retrieval: str = "hybrid",
        model_key: str | None = None,
        persona: str | None = None,
        task: str | None = None,
    ) -> RetrievalResult:
        intent_result = self.intent.predict(question)
        entities = self.entities.parse(question)

        query = build_query(intent_result.intent, entities)
        baseline_rows: List[Dict[str, object]] = []
        embed_rows: List[Dict[str, object]] = []

        client = KGClient(self.settings)
        try:
            if query and retrieval in ("baseline", "hybrid"):
                baseline_rows = client.run_query(query["text"], query.get("params"))
            if retrieval in ("embeddings", "hybrid"):
                embed_rows = self.embeddings.semantic_search(client, query=question, top_k=8)
        finally:
            client.close()

        context_parts: List[str] = []
        if baseline_rows:
            context_parts.append(f"Baseline rows: {baseline_rows}")
        if embed_rows:
            context_parts.append(f"Embedding hits: {embed_rows}")
        if not context_parts:
            context_parts.append("No results found in graph.")
        context = "\\n".join(context_parts)

        chosen_model = model_key or next(iter(self.llm_registry.options().keys()), None)
        model = self.llm_registry.get(chosen_model)
        answer = run_llm(
            model=model,
            context=context,
            persona=persona or self.settings.persona,
            task=task or self.settings.default_task,
            question=question,
        )

        return RetrievalResult(
            intent=intent_result.intent,
            entities=entities,
            cypher=query["text"] if query else None,
            params=query.get("params") if query else {},
            baseline_rows=baseline_rows,
            embed_rows=embed_rows,
            answer=answer,
        )

    def to_dict(self, result: RetrievalResult) -> Dict[str, object]:
        payload = asdict(result)
        payload["entities"] = result.entities.to_params()
        return payload
