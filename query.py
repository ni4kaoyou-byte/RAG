from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag_core import generate_answer, search_qdrant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query Qdrant collection and generate answer.")
    parser.add_argument("--query", required=True, help="Question for retrieval")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        help="OpenAI model name for generation (default: OPENAI_MODEL or gpt-5-mini)",
    )
    parser.add_argument("--qdrant-path", default="./qdrant_data", help="Qdrant local DB directory")
    parser.add_argument("--collection", default="rag_practice", help="Qdrant collection name")
    parser.add_argument("--no-generate", action="store_true", help="Only retrieval, skip LLM answer")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    client = QdrantClient(path=args.qdrant_path)

    if not client.collection_exists(args.collection):
        raise ValueError(
            f"Collection '{args.collection}' does not exist. "
            "Run ingest.py first."
        )
    count = client.count(collection_name=args.collection, exact=True).count
    if count == 0:
        raise ValueError(
            f"Collection '{args.collection}' is empty. "
            "Run ingest.py with --reindex."
        )

    model = SentenceTransformer(args.embedding_model)
    scores, hits = search_qdrant(client, args.collection, model, args.query, args.top_k)

    print("\n=== Retrieval Results ===")
    for i, (score, hit) in enumerate(zip(scores, hits), start=1):
        print(f"\n[{i}] score={score:.4f}")
        print(hit)

    if args.no_generate:
        return

    print("\n=== Generated Answer ===")
    print(generate_answer(args.query, hits, args.llm_model, os.getenv("OPENAI_API_KEY")))


if __name__ == "__main__":
    main()
