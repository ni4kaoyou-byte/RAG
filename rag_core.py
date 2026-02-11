from __future__ import annotations

from typing import Iterable

import numpy as np
from datasets import load_dataset
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


def load_and_chunk_dataset(
    dataset_name: str,
    split: str,
    text_column: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    dataset = load_dataset(dataset_name, split=split)
    texts = to_texts(dataset, text_column)
    if not texts:
        raise ValueError(
            f"No text found in column '{text_column}'. "
            "Pass correct --text-column for your dataset."
        )

    chunks: list[str] = []
    for text in texts:
        chunks.extend(chunk_text(text, chunk_size=chunk_size, overlap=overlap))

    if not chunks:
        raise ValueError("No chunks were created. Check chunk settings.")
    return chunks


def encode_normalized(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def recreate_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def upsert_text_points(
    client: QdrantClient,
    collection_name: str,
    vectors: np.ndarray,
    texts: list[str],
    batch_size: int = 256,
) -> None:
    point_id = 0
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_vectors = vectors[start : start + batch_size]
        points = []
        for vector, text in zip(batch_vectors, batch_texts):
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={"text": text},
                )
            )
            point_id += 1
        client.upsert(collection_name=collection_name, points=points)


def search_qdrant(
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


def generate_answer(query: str, contexts: list[str], model_name: str, api_key: str | None) -> str:
    if not api_key:
        return "OPENAI_API_KEY is not set. Skipping answer generation."
    if not contexts:
        return "No retrieved context. Skipping answer generation."

    client = OpenAI(api_key=api_key)
    context_block = "\n\n".join([f"[{i + 1}] {c}" for i, c in enumerate(contexts)])
    prompt = (
        "Use only the context below to answer.\n"
        "If the answer is not in context, say you do not know.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context_block}"
    )
    response = client.responses.create(model=model_name, input=prompt)
    return response.output_text.strip()
