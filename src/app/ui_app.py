import pathlib
import sys
import time

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

# Session state to hold past runs for comparison and easy clearing
if "runs" not in st.session_state:
    st.session_state["runs"] = []

with st.sidebar:
    st.header("Run settings")
    retrieval_primary = st.selectbox("Retrieval strategy", ["hybrid", "baseline", "embeddings"])
    retrieval_secondary = st.selectbox(
        "Compare with retrieval",
        ["None", "hybrid", "baseline", "embeddings"],
        help="Optional: run a second retrieval strategy for side-by-side comparison.",
    )
    retrieval_choices = [retrieval_primary]
    if retrieval_secondary != "None" and retrieval_secondary != retrieval_primary:
        retrieval_choices.append(retrieval_secondary)
    
    # Embedding model selector (when embeddings are enabled)
    embed_model_key = None
    if any(r in ("embeddings", "hybrid") for r in retrieval_choices):
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
    model_key_primary = st.selectbox("LLM model", model_options)
    model_key_secondary = st.selectbox(
        "Compare with LLM",
        ["None"] + model_options,
        index=0,
        help="Optional: run a second LLM for side-by-side comparison.",
    )
    model_choices = [model_key_primary]
    if model_key_secondary != "None" and model_key_secondary != model_key_primary:
        model_choices.append(model_key_secondary)

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
    if st.button("Clear results"):
        st.session_state["runs"] = []
        st.experimental_rerun()

question = st.text_input("Ask a question", placeholder="e.g., Best perfumes in SP with rating > 4?")
run = st.button("Run")

if run and question:
    new_runs = []
    for r_choice in retrieval_choices:
        for m_choice in model_choices:
            start = time.time()
            try:
                result = pipeline.run(
                    question=question,
                    retrieval=r_choice,
                    model_key=m_choice,
                    embed_model_key=embed_model_key,
                    persona=persona,
                    task=task,
                )
                duration = time.time() - start
                new_runs.append(
                    {
                        "retrieval": r_choice,
                        "model": m_choice,
                        "result": result,
                        "duration": duration,
                        "error": None,
                    }
                )
            except Exception as e:
                duration = time.time() - start
                new_runs.append(
                    {
                        "retrieval": r_choice,
                        "model": m_choice,
                        "result": None,
                        "duration": duration,
                        "error": str(e),
                    }
                )
    st.session_state["runs"] = new_runs

if st.session_state.get("runs"):
    tabs = st.tabs(
        [f"{run['retrieval']} | {run['model']}" for run in st.session_state["runs"]]
    )
    for idx, (tab, run_info) in enumerate(zip(tabs, st.session_state["runs"])):
        with tab:
            st.subheader("Run stats")
            st.write(
                {
                    "retrieval": run_info["retrieval"],
                    "model": run_info["model"],
                    "duration_sec": round(run_info["duration"], 3),
                }
            )
            if run_info["error"]:
                st.error(f"Run failed: {run_info['error']}")
                continue
            result = run_info["result"]
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

            st.subheader("Answer")
            st.write(result.answer)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Baseline Rows ({len(result.baseline_rows)})")
                with st.expander("View baseline rows", expanded=False):
                    st.json(result.baseline_rows or [])
            with col2:
                st.subheader(f"Embedding Hits ({len(result.embed_rows)})")
                if result.embed_model_used:
                    st.caption(f"Using: {result.embed_model_used}")
                with st.expander("View embedding hits", expanded=False):
                    st.json(result.embed_rows or [])

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
                st.plotly_chart(fig, width="stretch", key=f"plot-{idx}")
            else:
                st.caption("No baseline rows to visualize.")

st.markdown(
    "Tip: switch retrieval/embedding/model to compare grounded answers. Inspect raw rows to verify hallucination control."
)
