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
    "and compare models."
)

with st.sidebar:
    st.header("Run settings")
    retrieval = st.selectbox("Retrieval strategy", ["hybrid", "baseline", "embeddings"])
    model_options = list(pipeline.llm_registry.options().keys())
    if not model_options:
        st.error("No LLMs configured. Set OPENAI_API_KEY, OLLAMA_MODEL, or HUGGINGFACEHUB_API_TOKEN.")
        st.stop()
    model_key = st.selectbox("LLM model", model_options)
    persona = st.text_area("Persona", settings.persona, height=100)
    task = st.text_area("Task", settings.default_task, height=80)
    st.write("Environment")
    st.code(
        f"NEO4J_URI={settings.neo4j_uri}\nDB={settings.neo4j_database}\nVECTOR_INDEX={settings.vector_index}\nEMBED_MODEL={settings.embed_model}",
        language="bash",
    )

question = st.text_input("Ask a question", placeholder="e.g., Best perfumes in SP with rating > 4?")
run = st.button("Run")

if run and question:
    with st.spinner("Running pipeline..."):
        result = pipeline.run(
            question=question,
            retrieval=retrieval,
            model_key=model_key,
            persona=persona,
            task=task,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Intent & Entities")
        st.json(
            {
                "intent": result.intent,
                "entities": result.entities.to_params(),
                "cypher": result.cypher,
                "params": result.params,
            }
        )
        st.subheader("Baseline Rows")
        st.json(result.baseline_rows or "None")
    with col2:
        st.subheader("Embedding Hits")
        st.json(result.embed_rows or "None")
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

st.markdown("Tip: switch retrieval/model to compare grounded answers. Inspect raw rows to verify hallucination control.")
