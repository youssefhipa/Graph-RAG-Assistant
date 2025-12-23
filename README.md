# Graph-RAG Ecommerce Assistant (Milestone 3)

End-to-end Graph-RAG assistant for the ecommerce/marketplace theme. The system grounds LLM answers on a Neo4j knowledge graph, supports baseline Cypher retrieval + embedding similarity, compares multiple LLMs, and exposes a Streamlit UI.

## Quick start
- Create and activate a Python 3.10+ virtualenv.
- Install deps: `pip install -r requirements.txt`
- Run Neo4j (5.x or Aura) and set environment variables (see **Configuration**).
- Load your data: adapt `data/sample_import.cypher` or import the Milestone 2 CSVs into Neo4j.
- Export `PYTHONPATH=src` (or run commands from repo root so `src` is discoverable).
- Launch the UI: `streamlit run src/app/ui_app.py`

## Project layout
- `src/app/config.py` — env-driven settings (Neo4j, embeddings, LLMs, persona defaults).
- `src/app/intent.py` — rule-based intent classifier for ecommerce intents.
- `src/app/entities.py` — lightweight entity extraction for categories, states, cities, dates, ratings.
- `src/app/queries.py` — library of 10+ Cypher templates + parameter builder.
- `src/app/kg_client.py` — Neo4j driver helper to run Cypher & vector queries.
- `src/app/embedding.py` — embedding helper (SentenceTransformers by default) + Neo4j vector search.
- `src/app/llm.py` — registry for multiple chat models (OpenAI, Ollama; optional Hugging Face endpoint).
- `src/app/pipeline.py` — orchestrates: preprocess → retrieve (baseline + embeddings) → prompt → LLM.
- `src/app/ui_app.py` — Streamlit UI with model/retrieval selectors and transparency panes.
- `data/sample_import.cypher` — tiny sample KG seed to validate the pipeline.
- `tests/test_intent.py` — smoke test for intent classifier coverage.

## Configuration
Set these env vars or create a `.env` file:
```
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
VECTOR_INDEX=product_feature_index
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
OPENAI_API_KEY=...
HUGGINGFACEHUB_API_TOKEN=...  # leave blank if you don't use Hugging Face
OLLAMA_MODEL=llama2
```
You can swap embedding/LLM models via the UI dropdowns. The embedding model must match whatever you used to precompute vectors in the KG.
Set at least one LLM backend (OpenAI or Ollama). Hugging Face is optional; leave the token unset if you don't use it.

## Baseline Cypher library (examples)
- Product search by category/state/city/rating.
- Top sellers by rating or reliability.
- Delivery delays by state and the Delivery Impact Rule (late → bad review).
- Review sentiment + representative reviews.
- Category-level stats and trending categories.
- Customer behavior (repeat buyers) and state-level trends.
- Seller performance with on-time rate.
Queries are templated in `src/app/queries.py`; parameters are filled from extracted entities.

## Embedding retrieval
- Uses SentenceTransformers (default) to embed the user query.
- Queries a Neo4j vector index via `db.index.vector.queryNodes`.
- Works with node embeddings or feature-string embeddings; you choose the property (`embedding`) and index name via settings.
- To build the index (example):
  1. Compute embeddings for your products/features and store them on the nodes under the `embedding` property (matching `EMBED_PROPERTY`).
  2. Create the index in Neo4j:
     ```cypher
     CREATE VECTOR INDEX product_feature_index IF NOT EXISTS
     FOR (p:Product) ON (p.embedding)
     OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: "cosine"}};
     ```
  3. Change `EMBED_MODEL` and rerun to compare at least two embedding models.

## LLM layer & comparison
- Unified prompt structure: **context** (retrieval results) + **persona** (assistant role) + **task** (grounded answer).
- Registry supports OpenAI (gpt-3.5/4) and Ollama/local by default. A Hugging Face Inference endpoint is available but optional; leave the token unset to disable it. Add more in `src/app/llm.py`.
- `pipeline.py` returns raw context + final answer so you can log tokens, latency, and subjective quality. Fill the `MODEL_COMPARISON` table in your report.

## UI (Streamlit)
- Input box for the question, selectors for retrieval method (baseline / embeddings / hybrid) and model.
- Shows executed Cypher queries, baseline rows, embedding hits, and the grounded final answer.
- Optional graph preview using Plotly/NetworkX for the retrieved subgraph.

## Experiments & reporting
Document in your slides/report:
- System architecture diagram (preprocess → retrieval → LLM → UI).
- Retrieval examples (Cypher and embedding outputs).
- Model comparison (quantitative: latency/tokens/cost; qualitative: relevance/factuality).
- Error analysis + improvements + remaining limitations.

## Running tests
```
pytest -q
```
Tests are lightweight and offline.

## Data & embeddings
- Neo4j schema assumed: `Product`, `Order`, `OrderItem`, `Customer`, `Review` with relationships `REFERS_TO`, `CONTAINS`, `PLACED`, `REVIEWS`. Queries use properties like `product_category_name`, `price`, `customer_state`, `customer_city`, `review_score`.
- Embeddings: Model 1 only (MiniLM). Keep `.env` with `EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2`, `EMBED_PROPERTY=embedding`, `VECTOR_INDEX=product_feature_index` (384 dims). Model 2 is disabled by leaving `EMBED_MODEL_2`/`VECTOR_INDEX_2` blank.
- To rebuild embeddings: run `python run.py` to write vectors to `Product.embedding`; then create the index:
  ```cypher
  CREATE VECTOR INDEX product_feature_index IF NOT EXISTS
  FOR (p:Product) ON (p.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: "cosine"}};
  ```
- Data hygiene: trim/lowercase category/city/state, cast numerics, standardize dates (ISO). Regenerate embeddings after normalization.
- Translation/normalization: non-English fields (e.g., `product_category_name`, city/state names) should be translated/standardized to English before use; the current pipeline assumes data is already pretranslated/normalized.

## Usage notes
- Set env in `.env`: `NEO4J_URI=bolt://localhost:7687`, `NEO4J_USER/NEO4J_PASSWORD`; optionally add `OPENAI_API_KEY`. Ollama models are fixed in code (llama2, phi3:mini, mistral); no env needed. Hugging Face is disabled unless you set `HUGGINGFACEHUB_API_TOKEN`. Leave Model 2 vars empty.
- Install deps: `pip install -r requirements.txt`. Add `export PYTHONPATH=src` (or `pip install -e .`).
- Run UI: `streamlit run src/app/ui_app.py`. Choose retrieval (baseline/hybrid/embeddings) and LLM (OpenAI or Ollama; Hugging Face only appears if you set its token). Hybrid recommended.
- If schema differs, align `src/app/queries.py` to your labels/properties.

## Manual test script (run baseline vs hybrid; OpenAI vs Ollama)
- product_search: “Top electronics in SP with rating >4?”
- delivery_delay: “Which orders in RJ are late this month?”
- review_sentiment: “Reviews for electronics in sao paulo?”
- seller_performance: “Best sellers in SP by reliability >0.8?”
- state_trend: “Which state has most orders?”
- category_insight: “Most popular product categories?”
- recommendation: “Recommend perfumes in RJ rating >4.”
- customer_behavior: “Customers with repeat orders?”
- seller_count: “How many sellers are there?”
- faq/unknown: “What is the return policy?” (expect graceful no-data)
Record model, retrieval, latency, baseline rows count, embedding hits count, and whether the answer is grounded in returned context.
