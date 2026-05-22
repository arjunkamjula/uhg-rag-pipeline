"""
retrieval/vector_search.py

Pinecone vector similarity search with metadata filtering.

Embeds the query using the same model as ingestion (all-MiniLM-L6-v2)
so both live in the same 384-dim vector space — which is why the search works.
Applies optional metadata filters (member_id, doc_type, claim_id) simultaneously
with the cosine similarity search using Pinecone's pre-filtering.
"""

import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX_NAME", "uhg-claims-rag")
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

_embedding_model = None
_pinecone_index  = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(PINECONE_INDEX)
    return _pinecone_index


def search(
    question:  str,
    top_k:     int  = 4,
    member_id: str  = None,
    doc_type:  str  = None,
    claim_id:  str  = None,
) -> list:
    model = get_embedding_model()
    index = get_pinecone_index()

    query_vector = model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    filters = {}
    if member_id:
        filters["member_id"] = {"$eq": member_id}
    if doc_type:
        filters["doc_type"]  = {"$eq": doc_type}
    if claim_id:
        filters["claim_id"]  = {"$eq": claim_id}

    query_kwargs = {
        "vector":           query_vector,
        "top_k":            top_k,
        "include_metadata": True,
    }
    if filters:
        query_kwargs["filter"] = filters

    response = index.query(**query_kwargs)

    results = []
    for match in response.matches:
        meta = match.metadata or {}
        results.append({
            "text":        meta.get("chunk_text", ""),
            "score":       round(float(match.score), 4),
            "doc_type":    meta.get("doc_type", "unknown"),
            "member_id":   meta.get("member_id"),
            "claim_id":    meta.get("claim_id"),
            "auth_id":     meta.get("auth_id"),
            "source_file": meta.get("source_file", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "date":        meta.get("date"),
        })

    return results


def search_by_doc_types(
    question:  str,
    doc_types: list,
    member_id: str = None,
    top_k:     int = 4,
) -> list:
    model = get_embedding_model()
    index = get_pinecone_index()

    query_vector = model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    filters = {"doc_type": {"$in": doc_types}}
    if member_id:
        filters["member_id"] = {"$eq": member_id}

    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=filters,
    )

    results = []
    for match in response.matches:
        meta = match.metadata or {}
        results.append({
            "text":        meta.get("chunk_text", ""),
            "score":       round(float(match.score), 4),
            "doc_type":    meta.get("doc_type", "unknown"),
            "member_id":   meta.get("member_id"),
            "claim_id":    meta.get("claim_id"),
            "source_file": meta.get("source_file", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "date":        meta.get("date"),
        })

    return results
