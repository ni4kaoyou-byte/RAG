from __future__ import annotations

import argparse
import os
import subprocess
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG practice runner (ingest + query).")
    parser.add_argument("--dataset", default="ag_news", help="Hugging Face dataset name")
    parser.add_argument("--split", default="train[:200]", help="Dataset split")
    parser.add_argument("--text-column", default="text", help="Column used as source text")
    parser.add_argument("--query", required=True, help="Question for retrieval")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=100, help="Chunk overlap in characters")
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
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Delete and rebuild collection before query",
    )
    return parser.parse_args()


def run_ingest(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "ingest.py",
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--text-column",
        args.text_column,
        "--chunk-size",
        str(args.chunk_size),
        "--overlap",
        str(args.overlap),
        "--embedding-model",
        args.embedding_model,
        "--qdrant-path",
        args.qdrant_path,
        "--collection",
        args.collection,
    ]
    if args.reindex:
        cmd.append("--reindex")
    subprocess.run(cmd, check=True)


def run_query(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "query.py",
        "--query",
        args.query,
        "--top-k",
        str(args.top_k),
        "--embedding-model",
        args.embedding_model,
        "--llm-model",
        args.llm_model,
        "--qdrant-path",
        args.qdrant_path,
        "--collection",
        args.collection,
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    load_dotenv()
    args = parse_args()
    client = QdrantClient(path=args.qdrant_path)
    needs_ingest = args.reindex or not client.collection_exists(args.collection)
    if needs_ingest:
        run_ingest(args)
    run_query(args)


if __name__ == "__main__":
    main()
