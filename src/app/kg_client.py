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
        """Execute a Cypher query with parameters, handling NULL params gracefully."""
        params = params or {}
        with self.driver.session(database=self.settings.neo4j_database) as session:
            result = session.run(query, **params)
            return [record.data() for record in result]

    def vector_query(
        self,
        vector: List[float],
        top_k: int = 10,
        index_name: str | None = None,
        embed_property: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Query a Neo4j vector index with validation.
        
        Args:
            vector: The embedding vector to search with.
            top_k: Number of top results to return.
            index_name: Name of the vector index. Uses settings.vector_index if not provided.
            embed_property: Node property to null out in the return (avoids sending big vectors back).
            
        Returns:
            List of matching records with similarity scores.
            
        Raises:
            ValueError: If vector dimensions or index parameters are invalid.
            RuntimeError: If vector query execution fails.
        """
        # Validate input parameters
        if not vector:
            raise ValueError("Vector cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        
        index_name = index_name or self.settings.vector_index
        embed_property = embed_property or self.settings.embed_property
        
        # Validate index name (basic sanity check)
        if not index_name or not isinstance(index_name, str):
            raise ValueError(f"Invalid index name: {index_name}")
        
        try:
            cypher = f"""
            CALL db.index.vector.queryNodes(
                '{index_name}',
                $top_k,
                $vector
            ) YIELD node, score
            RETURN node{{.*, `{embed_property}`: null}} AS item, score
            ORDER BY score DESC
            """
            with self.driver.session(database=self.settings.neo4j_database) as session:
                result = session.run(cypher, vector=vector, top_k=top_k)
                records = [record.data() for record in result]
                
                if not records:
                    print(f"Vector search returned no results for index '{index_name}'")
                
                return records
        except Exception as e:
            error_msg = f"Vector query failed on index '{index_name}': {str(e)}"
            print(f"Error: {error_msg}")
            raise RuntimeError(error_msg) from e
