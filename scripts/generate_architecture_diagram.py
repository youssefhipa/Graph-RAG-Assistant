"""
Generate a simple architecture diagram for the Graph-RAG Ecommerce Assistant.

Run from repo root:
    python scripts/generate_architecture_diagram.py

Outputs:
    architecture.png in the project root.
"""

import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle


def add_box(ax, xy, text, width=1.8, height=0.8, color="#1f77b4"):
    x, y = xy
    rect = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        edgecolor="black",
        facecolor=color,
        alpha=0.15,
    )
    ax.add_patch(rect)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        weight="semibold",
    )
    return rect


def add_arrow(ax, start, end, text=None):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle=ArrowStyle("-|>", head_length=6, head_width=4), color="black", lw=1.0),
    )
    if text:
        midx = (start[0] + end[0]) / 2
        midy = (start[1] + end[1]) / 2
        ax.text(midx, midy + 0.05, text, ha="center", va="bottom", fontsize=9)


def main():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # Row positions
    y_top = 4.5
    y_mid = 3
    y_retrieve = 1.5
    y_data = 0.3

    # UI
    ui = add_box(ax, (0.5, y_top), "Streamlit UI\n(user input)", width=2.2)

    # Pipeline
    pipeline = add_box(ax, (3.0, y_top), "Pipeline\n(intent + entities)", width=2.2, color="#2ca02c")
    prompt = add_box(ax, (5.5, y_top), "Prompt\nbuilder", width=1.8, color="#2ca02c")

    # Retrieval
    baseline = add_box(ax, (3.0, y_retrieve), "Baseline\nCypher", width=1.8, color="#ff7f0e")
    embed = add_box(ax, (5.2, y_retrieve), "Embedding\nsearch", width=1.8, color="#ff7f0e")

    # LLMs
    llm = add_box(ax, (7.5, y_top), "LLM\n(HF / Ollama)", width=2.0, color="#d62728")

    # Data
    kg = add_box(ax, (4.1, y_data), "Neo4j\n(Product/Order/...)\nVector index", width=2.5, height=1.0, color="#9467bd")

    # Arrows
    add_arrow(ax, (2.7, y_top + 0.4), (3.0, y_top + 0.4), "text")
    add_arrow(ax, (4.2, y_top + 0.4), (5.5, y_top + 0.4))
    add_arrow(ax, (7.3, y_top + 0.4), (7.5, y_top + 0.4), "prompt")
    add_arrow(ax, (9.5, y_top + 0.4), (9.5, y_top - 0.4))
    add_arrow(ax, (9.5, y_top - 0.4), (2.0, y_top - 0.4), "answer")

    add_arrow(ax, (3.9, y_top), (3.9, y_retrieve + 0.8), "intent/entities")
    add_arrow(ax, (6.0, y_top), (6.0, y_retrieve + 0.8), "context need")

    add_arrow(ax, (3.9, y_retrieve + 0.1), (4.0, y_data + 1.0))
    add_arrow(ax, (6.0, y_retrieve + 0.1), (5.8, y_data + 1.0))

    add_arrow(ax, (4.4, y_data + 1.0), (4.1, y_retrieve + 0.1))
    add_arrow(ax, (5.6, y_data + 1.0), (5.2, y_retrieve + 0.1))

    add_arrow(ax, (3.9, y_retrieve + 0.8), (5.6, y_top), "context")
    add_arrow(ax, (5.0, y_retrieve + 0.8), (5.6, y_top))

    out_path = pathlib.Path(__file__).resolve().parents[1] / "architecture.png"
    fig.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved diagram to {out_path}")


if __name__ == "__main__":
    main()
