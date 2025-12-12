import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    neo4j_uri: str = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    vector_index: str = os.getenv("VECTOR_INDEX", "product_feature_index")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embed_property: str = os.getenv("EMBED_PROPERTY", "embedding")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    huggingface_token: str | None = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    ollama_model: str | None = os.getenv("OLLAMA_MODEL")
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


def get_settings() -> Settings:
    return Settings()
