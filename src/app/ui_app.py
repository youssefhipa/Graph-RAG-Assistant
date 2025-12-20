import pathlib
import sys

# Ensure src/ is on sys.path when running via `streamlit run`
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402


st.set_page_config(page_title="Graph-RAG Ecommerce Assistant", layout="wide")
settings = get_settings()
pipeline = Pipeline()

st.title("Graph-RAG Ecommerce Assistant")
st.markdown(
    "Ground answers on the Neo4j knowledge graph. Switch between baseline Cypher and embeddings, "
    "compare embedding models, and compare LLMs."
)

with st.sidebar:
    st.header("Run settings")
    retrieval = st.selectbox("Retrieval strategy", ["hybrid", "baseline", "embeddings"])
    
    # Embedding model selector (when embeddings are enabled)
    embed_model_key = None
    if retrieval in ("embeddings", "hybrid"):
        embedding_models = settings.get_embedding_models()
        embed_options = list(embedding_models.keys())
        if embed_options:
            embed_model_key = st.selectbox(
                "Embedding model",
                embed_options,
                format_func=lambda k: embedding_models[k].name,
                help="Select which embedding model to use for vector search."
            )
    
    model_options = list(pipeline.llm_registry.options().keys())
    if not model_options:
        st.error("No LLMs configured. Set OPENAI_API_KEY, OLLAMA_MODEL, or HUGGINGFACEHUB_API_TOKEN.")
        st.stop()
    model_key = st.selectbox("LLM model", model_options)
    persona = st.text_area("Persona", settings.persona, height=100)
    task = st.text_area("Task", settings.default_task, height=80)
    st.write("Environment")
    env_text = f"""NEO4J_URI={settings.neo4j_uri}
DB={settings.neo4j_database}
VECTOR_INDEX={settings.vector_index}
EMBED_MODEL={settings.embed_model}"""
    if settings.embed_model_2:
        env_text += f"\nVECTOR_INDEX_2={settings.vector_index_2}\nEMBED_MODEL_2={settings.embed_model_2}"
    st.code(env_text, language="bash")

question = st.text_input("Ask a question", placeholder="e.g., Best perfumes in SP with rating > 4?")
run = st.button("Run")

if run and question:
    with st.spinner("Running pipeline..."):
        result = pipeline.run(
            question=question,
            retrieval=retrieval,
            model_key=model_key,
            embed_model_key=embed_model_key,
            persona=persona,
            task=task,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Intent & Entities")
        intent_info = {
            "intent": result.intent,
            "entities": result.entities.to_params(),
            "cypher": result.cypher,
            "params": result.params,
        }
        if result.embed_model_used:
            intent_info["embedding_model"] = result.embed_model_used
        st.json(intent_info)
        st.subheader("Baseline Rows")
        st.json(result.baseline_rows or [])
    with col2:
        st.subheader("Embedding Hits")
        if result.embed_model_used:
            st.caption(f"Using: {result.embed_model_used}")
        st.json(result.embed_rows or [])
        st.subheader("Answer")
        st.write(result.answer)

    st.subheader("Graph preview")
    if result.baseline_rows:
        headers = list(result.baseline_rows[0].keys())
        columns = [[row.get(h) for row in result.baseline_rows] for h in headers]
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=headers),
                    cells=dict(values=columns),
                )
            ]
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("Tip: switch retrieval/embedding/model to compare grounded answers. Inspect raw rows to verify hallucination control.")