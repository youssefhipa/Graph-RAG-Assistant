# Presentation Content — Graph-RAG Ecommerce Assistant (Milestone 3)

## Slide 1 — Title & Goal
- Title: Graph-RAG Ecommerce Assistant (Milestone 3)
- Goal: Provide grounded Q&A on an ecommerce Neo4j graph using baseline Cypher, embedding retrieval, and multiple LLMs (Ollama models; HF optional).

## Slide 2 — Architecture
- Flow: Streamlit UI → Pipeline (intent + entity extraction) → Retrieval (baseline Cypher + embedding search) → Prompt builder → LLM (Ollama) → Answer.
- Data: Neo4j graph (Product, Order, OrderItem, Customer, Review) with vector index on `Product.embedding` (384-d MiniLM).

## Slide 3 — Data & Schema
- Nodes/Relationships: Product, Order, OrderItem, Customer, Review with REFERS_TO, CONTAINS, PLACED, REVIEWS.
- Key properties: `product_category_name`, `price`, `customer_state`, `customer_city`, `review_score`, `embedding`.
- Vector index: `product_feature_index` on `Product.embedding` (384 dims).
- Note: Data assumed translated/normalized to English (categories, city/state).

## Slide 4 — Config & Runtime
- `.env`: `NEO4J_URI=bolt://localhost:7687`, `NEO4J_USER/NEO4J_PASSWORD`. HF token optional; Model 2 optional (set `_2` vars if using `embedding2` + `product_feature_index_2`).
- Install/run: `pip install -r requirements.txt`; set `PYTHONPATH=src` or `pip install -e .`; start UI with `streamlit run src/app/ui_app.py`.

## Slide 5 — Retrieval Logic
- Baseline Cypher in `queries.py` aligned to current schema (10+ templates).
- Embedding search: Model 1 (MiniLM) on `product_feature_index`; Model 2 available via `embedding2` + `product_feature_index_2` for comparison.
- Hybrid mode merges baseline rows + embedding hits; raw context shown in UI.

## Slide 6 — LLM Layer
- Ollama via `langchain-ollama` with fixed models (`ollama-llama2`, `ollama-phi3:mini`, `ollama-mistral`); HF optional.
- Prompt: persona + task + context + question; answers instructed to stay within provided context.

## Slide 7 — UI Walkthrough
- Sidebar: retrieval mode (baseline/hybrid/embeddings), embedding model (Model 1), LLM (HF/Ollama), persona/task editor, env display.
- Main: intent/entities view, baseline rows, embedding hits, final answer, graph preview, run stats/tabs for comparisons.

## Slide 8 — Retrieval Examples
- Example (product_search hybrid): show Cypher text, baseline rows (id/category/price/state/city/rating), embedding hits with scores, grounded answer snippet.
- Baseline vs hybrid comparison to illustrate recall improvement.

## Slide 9 — Testing & Error Handling
- Manual test prompts (baseline vs hybrid; compare Ollama models): product_search, delivery_delay, review_sentiment, seller_performance, state_trend, category_insight, recommendation, customer_behavior, seller_count, faq.
- Fixes: schema mismatch resolved in queries; Ollama deprecation fixed; Model 2 available but optional; baseline params always bound to avoid ParameterMissing.

## Slide 10 — Limitations & Next Steps
- Limitations: single embedding model active; regex-based intent/entities; ratings may be sparse; limited logging/guardrails.
- Next: richer NER/intent, enable second embedding property/index if needed, add logging/metrics/caching, add empty-context guardrail, expand eval harness.
