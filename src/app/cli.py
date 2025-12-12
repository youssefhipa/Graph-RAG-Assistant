import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.pipeline import Pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Graph-RAG pipeline once.")
    parser.add_argument("question", help="User question to answer")
    parser.add_argument("--retrieval", choices=["baseline", "embeddings", "hybrid"], default="hybrid")
    parser.add_argument("--model", dest="model", default=None)
    args = parser.parse_args()

    pipe = Pipeline()
    result = pipe.run(question=args.question, retrieval=args.retrieval, model_key=args.model)
    print("Intent:", result.intent)
    print("Entities:", result.entities.to_params())
    print("Cypher:", result.cypher)
    print("Baseline rows:", result.baseline_rows)
    print("Embedding rows:", result.embed_rows)
    print("Answer:", result.answer)


if __name__ == "__main__":
    main()
