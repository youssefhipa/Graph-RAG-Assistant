# Manual Testing Guide

## Setup

1. **Configure environment variables** in `.env`:
```bash
# Neo4j connection
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Vector indices and embedding models
VECTOR_INDEX=product_feature_index
VECTOR_INDEX_2=product_feature_index_2
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBED_MODEL_2=sentence-transformers/all-mpnet-base-v2

# LLM configuration (choose at least one)
OPENAI_API_KEY=your_key
# OR
OLLAMA_MODEL=llama2
# OR
HUGGINGFACEHUB_API_TOKEN=your_token