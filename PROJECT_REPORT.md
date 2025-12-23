# Project Report — Graph-RAG Ecommerce Assistant

## Status
- Current configuration runs with Model 1 embeddings (MiniLM) and three fixed Ollama models (`ollama-llama2`, `ollama-phi3:mini`, `ollama-mistral`). Model 2 can be enabled by setting `EMBED_MODEL_2`/`EMBED_PROPERTY_2`/`VECTOR_INDEX_2` (embedding2 + product_feature_index_2). HuggingFace is optional.
- Neo4j schema aligned: `Product`, `Order`, `OrderItem`, `Customer`, `Review` with relationships `REFERS_TO`, `CONTAINS`, `PLACED`, `REVIEWS`. Queries use properties `product_category_name`, `price`, `customer_state`, `customer_city`, `review_score`.
- Vector index in use: `product_feature_index` on `Product.embedding` (384 dims).

## Configuration
- `.env` keys: `NEO4J_URI=bolt://localhost:7687`, `NEO4J_USER/NEO4J_PASSWORD`. HuggingFace token is optional (not needed). Model 2 vars remain empty.
- Install deps: `pip install -r requirements.txt`. Set `PYTHONPATH=src` (or `pip install -e .`).
- Run UI: `streamlit run src/app/ui_app.py`.

## Architecture
- Streamlit UI → Pipeline (intent + entity) → Retrieval: baseline Cypher (`kg_client.py`) + embedding search (`embedding.py`) → Prompt (`llm.py`) → LLM (HuggingFace/Ollama) → Answer.
- Data layer: Neo4j with vector index on `Product.embedding`.

## Retrieval logic
- Baseline Cypher (`queries.py`) adapted to the current schema for product_search and other intents.
- Embeddings: Model 1 (MiniLM) on `embedding` / `product_feature_index`. Hybrid mode combines both.

## LLMs
- Ollama via `langchain-ollama` with three fixed models: `ollama-llama2`, `ollama-phi3:mini`, `ollama-mistral` (ensure pulled).
- HuggingFace endpoint supported but optional.

## Compliance with milestone requirements
- Input preprocessing: intent classifier + entity extractor for ecommerce entities; optional translation/normalization assumed.
- Baseline retrieval: 10+ Cypher templates in `queries.py`, parameterized with extracted entities.
- Embeddings: Model 1 active; Model 2 available via `embedding2` + `product_feature_index_2` for comparison.
- LLM layer: three LLMs (Ollama models) selectable in UI; prompt structured as context/persona/task.
- UI: shows raw baseline rows, embedding hits, Cypher context, and final answer; model/retrieval selectors; graph preview.
- Experiments: supports baseline vs embeddings vs hybrid; LLM comparison via model dropdown.
- Reports/slides: architecture, retrieval examples, model comparison, error analysis, improvements/limitations captured.

## Data & indexes
- Embeddings stored at `Product.embedding`. Rebuild index if reloading data:
  ```cypher
  CREATE VECTOR INDEX product_feature_index IF NOT EXISTS
  FOR (p:Product) ON (p.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: "cosine"}};
  ```

## Manual test checklist (baseline vs hybrid; HF vs Ollama)
- product_search: “Top electronics in SP with rating >4?”
- delivery_delay: “Which orders in RJ are late this month?”
- review_sentiment: “Reviews for electronics in sao paulo?”
- seller_performance: “Best sellers in SP by reliability >0.8?”
- state_trend: “Which state has most orders?”
- category_insight: “Most popular product categories?”
- recommendation: “Recommend perfumes in RJ rating >4.”
- customer_behavior: “Customers with repeat orders?”
- seller_count: “How many sellers are there?”
- faq/unknown: “What is the return policy?” (expect no-data response)
Record: model, retrieval mode, latency, baseline row count, embedding hit count, groundedness.

## Known limitations
- Single embedding model active (Model 2 disabled).
- Intent/entity extraction is regex-based; could be upgraded.
- Ratings derived from reviews may be sparse.
