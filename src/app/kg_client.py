from __future__ import annotations

from typing import Any, Dict, List

from neo4j import GraphDatabase, basic_auth

from .config import Settings


class KGClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=basic_auth(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def run_query(self, query: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        params = params or {}
        with self.driver.session(database=self.settings.neo4j_database) as session:
            result = session.run(query, **params)
            return [record.data() for record in result]

    def vector_query(self, vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Query a Neo4j vector index. Ensure the index exists and that `embedding`
        property matches `settings.embed_property`.
        """
        cypher = f"""
        CALL db.index.vector.queryNodes(
            '{self.settings.vector_index}',
            $top_k,
            $query
        ) YIELD node, score
        RETURN node{{.*, `{self.settings.embed_property}`: null}} AS item, score
        ORDER BY score DESC
        """
        with self.driver.session(database=self.settings.neo4j_database) as session:
            result = session.run(cypher, query=vector, top_k=top_k)
            return [record.data() for record in result]
