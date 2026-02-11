from __future__ import annotations

import argparse

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag_core import encode_normalized, load_and_chunk_dataset, recreate_collection, upsert_text_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest dataset into Qdrant.")
    parser.add_argument("--dataset", default="ag_news", help="Hugging Face dataset name")
    parser.add_argument("--split", default="train[:200]", help="Dataset split")
    parser.add_argument("--text-column", default="text", help="Column used as source text")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=100, help="Chunk overlap in characters")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--qdrant-path", default="./qdrant_data", help="Qdrant local DB directory")
    parser.add_argument("--collection", default="rag_practice", help="Qdrant collection name")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Delete and rebuild collection even if it already exists",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    client = QdrantClient(path=args.qdrant_path)

    if client.collection_exists(args.collection) and not args.reindex:
        count = client.count(collection_name=args.collection, exact=True).count
        print(
            f"Collection '{args.collection}' already exists ({count} points). "
            "Skip ingestion. Use --reindex to rebuild."
        )
        return

    chunks = load_and_chunk_dataset(
        dataset_name=args.dataset,
        split=args.split,
        text_column=args.text_column,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    model = SentenceTransformer(args.embedding_model)
    vectors = encode_normalized(model, chunks)
    recreate_collection(client, args.collection, vectors.shape[1])
    upsert_text_points(client, args.collection, vectors, chunks)
    count = client.count(collection_name=args.collection, exact=True).count
    print(f"Ingest completed. Collection '{args.collection}' now has {count} points.")


if __name__ == "__main__":
    main()
