# Slide/Report Notes (Yassin)

## Architecture
- Streamlit UI → Pipeline (intent + entity) → Retrieval (baseline Cypher via `kg_client.py`, embedding search via `embedding.py` on Model 1 index) → Prompt (`llm.py`) → LLM (Ollama models) → Answer.
- Data: Neo4j graph with Product/Order/OrderItem/Customer/Review; vector index on `Product.embedding` (384-d MiniLM).

## Retrieval examples (show screenshots/logs)
- Product search hybrid run: Cypher text + rows (id, category, price, customer_state/city, rating). Embedding hits from `product_feature_index` with scores. Final answer citing context.
- Comparison: baseline vs hybrid for the same question to show added recall; HF vs Ollama answers/latency.

## Model comparison
- Three Ollama models: `ollama-llama2`, `ollama-phi3:mini`, `ollama-mistral` (Model 2 embeddings disabled).
- Record: latency, answer quality, hallucination rate; hybrid vs baseline-only.

## Error analysis
- Schema mismatch fixed: queries now use `product_category_name`, `customer_state/city`, `review_score` (no Category/BELONGS_TO).
- Ollama deprecation resolved with `langchain-ollama`; HF optional.
- Model 2 disabled due to single embedding property in use; avoid hitting missing index.

## Improvements / limitations
- Limitations: Single embedding model active; regex intent/entities; ratings derived from reviews and may be sparse; no caching/observability; no guard for empty context beyond prompt.
- Improvements: Add richer NER/intent (spaCy/LLM); add second embedding property/index if needed; add empty-context guardrail; add latency/error logging; add eval harness.

## Manual test script (baseline vs hybrid; HF vs Ollama)
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
For each: record model, retrieval, latency, baseline row count, embedding hit count, groundedness of answer.
