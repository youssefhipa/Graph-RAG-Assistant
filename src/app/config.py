import os
from dataclasses import dataclass
from typing import Optional, List, Dict

from dotenv import load_dotenv


load_dotenv()


@dataclass
class EmbeddingModelConfig:
    """Configuration for an embedding model."""
    name: str
    model_id: str
    vector_index: str
    embed_property: str


@dataclass
class Settings:
    neo4j_uri: str = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # Primary embedding model (legacy support)
    vector_index: str = os.getenv("VECTOR_INDEX", "product_feature_index")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embed_property: str = os.getenv("EMBED_PROPERTY", "embedding")
    
    # Secondary embedding model
    embed_model_2: Optional[str] = os.getenv("EMBED_MODEL_2", "sentence-transformers/all-mpnet-base-v2")
    vector_index_2: Optional[str] = os.getenv("VECTOR_INDEX_2", "product_feature_index_2")
    embed_property_2: Optional[str] = os.getenv("EMBED_PROPERTY_2", "embedding2")
    
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    huggingface_token: Optional[str] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    ollama_model: Optional[str] = os.getenv("OLLAMA_MODEL")
    persona: str = os.getenv(
        "ASSISTANT_PERSONA",
        "You are an intelligent ecommerce marketplace analyst. "
        "You answer with concise, factual insights grounded in the provided graph context.",
    )
    default_task: str = os.getenv(
        "ASSISTANT_TASK",
        "Use only the provided context to answer. "
        "If information is missing, say so and avoid hallucinations.",
    )

    def get_embedding_models(self) -> Dict[str, EmbeddingModelConfig]:
        """Return a dictionary of available embedding models."""
        models = {
            "model_1": EmbeddingModelConfig(
                name="Model 1 (Primary)",
                model_id=self.embed_model,
                vector_index=self.vector_index,
                embed_property=self.embed_property,
            )
        }
        if self.embed_model_2:
            models["model_2"] = EmbeddingModelConfig(
                name="Model 2 (Secondary)",
                model_id=self.embed_model_2,
                vector_index=self.vector_index_2 or self.vector_index,
                embed_property=self.embed_property_2 or self.embed_property,
            )
        return models


def get_settings() -> Settings:
    return Settings()
