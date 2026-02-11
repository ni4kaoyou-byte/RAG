from __future__ import annotations

import argparse
import os
from typing import Iterable

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size]
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(text):
            break
    return chunks


def to_texts(records: Iterable[dict], text_column: str) -> list[str]:
    texts: list[str] = []
    for row in records:
        value = row.get(text_column)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
    return texts


def build_index(
    model: SentenceTransformer, texts: list[str], qdrant_path: str, collection_name: str
) -> QdrantClient:
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    dim = embeddings.shape[1]
    client = QdrantClient(path=qdrant_path)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(id=i, vector=emb.tolist(), payload={"text": text})
        for i, (emb, text) in enumerate(zip(embeddings, texts))
    ]
    client.upsert(collection_name=collection_name, points=points)
    return client


def search(
    client: QdrantClient,
    collection_name: str,
    model: SentenceTransformer,
    query: str,
    top_k: int,
) -> tuple[np.ndarray, list[str]]:
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")[0]
    query_vec = query_vec / max(np.linalg.norm(query_vec), 1e-12)
    hits = client.query_points(
        collection_name=collection_name,
        query=query_vec.tolist(),
        limit=top_k,
    ).points
    scores = np.array([hit.score for hit in hits], dtype="float32")
    results = [str(hit.payload.get("text", "")) for hit in hits]
    return scores, results


def generate_answer(query: str, contexts: list[str], model_name: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY is not set. Skipping answer generation."

    client = OpenAI(api_key=api_key)
    context_block = "\n\n".join([f"[{i + 1}] {c}" for i, c in enumerate(contexts)])
    prompt = (
        "Use only the context below to answer.\n"
        "If the answer is not in context, say you do not know.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context_block}"
    )

    response = client.responses.create(
        model=model_name,
        input=prompt,
    )
    return response.output_text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal RAG practice with HF Datasets + Qdrant.")
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
    parser.add_argument(
        "--qdrant-path",
        default="./qdrant_data",
        help="Qdrant local DB directory",
    )
    parser.add_argument(
        "--collection",
        default="rag_practice",
        help="Qdrant collection name",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    dataset = load_dataset(args.dataset, split=args.split)
    texts = to_texts(dataset, args.text_column)
    if not texts:
        raise ValueError(
            f"No text found in column '{args.text_column}'. "
            "Pass correct --text-column for your dataset."
        )

    corpus: list[str] = []
    for text in texts:
        corpus.extend(chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap))

    if not corpus:
        raise ValueError("No chunks were created. Check chunk settings.")

    model = SentenceTransformer(args.embedding_model)
    client = build_index(model, corpus, args.qdrant_path, args.collection)
    scores, hits = search(client, args.collection, model, args.query, args.top_k)

    print("\n=== Retrieval Results ===")
    for i, (score, hit) in enumerate(zip(scores, hits), start=1):
        print(f"\n[{i}] score={score:.4f}")
        print(hit)

    print("\n=== Generated Answer ===")
    print(generate_answer(args.query, hits, args.llm_model))


if __name__ == "__main__":
    main()
