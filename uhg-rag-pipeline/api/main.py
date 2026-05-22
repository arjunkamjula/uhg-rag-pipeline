"""
api/main.py

UHG RAG Pipeline — FastAPI service

Endpoints:
  POST /query        — main query endpoint, routes to SQL or RAG automatically
  POST /ingest       — trigger ingestion pipeline
  GET  /claim/{id}   — direct structured claim lookup
  GET  /member/{id}  — direct structured member lookup
  GET  /health       — system health check for load balancer / k8s liveness probe
  GET  /metrics      — document counts and vector store stats
  GET  /docs         — auto-generated OpenAPI docs (FastAPI built-in)

Run:
    uvicorn api.main:app --reload --port 8000
"""

import os
import time
import logging
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.schemas import (
    QueryRequest, QueryResponse, SourceDocument,
    IngestRequest, IngestResponse,
    ClaimResponse, MemberResponse, HealthResponse,
)
from api.dependencies import get_embedding_model, get_pinecone_index, get_db_engine
from retrieval.router import detect_route, QueryRoute
from retrieval.sql_lookup import lookup_claim, lookup_member, lookup_auth
from retrieval.vector_search import search as vector_search
from retrieval.prompt_builder import build_prompt, format_sources
from retrieval.llm_client import generate

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
ENVIRONMENT  = os.getenv("ENVIRONMENT", "development")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting UHG RAG Pipeline API...")
    logger.info("Loading embedding model...")
    get_embedding_model()
    logger.info("Connecting to Pinecone...")
    get_pinecone_index()
    logger.info("Connecting to PostgreSQL...")
    get_db_engine()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("uhg-rag-queries")
    logger.info("API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title       = "UHG RAG Pipeline API",
    description = "Clinical claims document intelligence — RAG + SQL query service",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main query endpoint. Routes to SQL or RAG based on intent detection.

    SQL path  — exact ID lookup, no LLM, ~6ms
    RAG path  — semantic search + LLM generation, ~2300ms
    """
    start_time = time.time()

    route_info = detect_route(request.question)
    route      = route_info["route"]
    member_id  = request.member_id or route_info.get("member_id")
    claim_id   = route_info.get("claim_id")

    # if member_id was explicitly passed in the request body,
    # upgrade vague RAG to focused RAG so the metadata filter is applied
    if request.member_id and route == QueryRoute.RAG_VAGUE:
        route                    = QueryRoute.RAG_FOCUSED
        route_info["route"]      = QueryRoute.RAG_FOCUSED
        route_info["confidence"] = "high"
        route_info["member_id"]  = request.member_id

    logger.info(
        f"Query: '{request.question[:80]}' "
        f"-> route={route} member={member_id} claim={claim_id}"
    )

    if route == QueryRoute.SQL_CLAIM and claim_id:
        result = lookup_claim(claim_id)
        if not result["found"]:
            raise HTTPException(404, detail=f"Claim {claim_id} not found")
        return QueryResponse(
            answer            = _format_claim_answer(result),
            route             = route,
            confidence        = route_info["confidence"],
            sources           = [],
            model             = "sql",
            prompt_tokens     = 0,
            completion_tokens = 0,
            latency_ms        = _ms(start_time),
            member_id         = result.get("member_id"),
            claim_id          = claim_id,
        )

    if route == QueryRoute.SQL_MEMBER and member_id:
        result = lookup_member(member_id)
        if not result["found"]:
            raise HTTPException(404, detail=f"Member {member_id} not found")
        return QueryResponse(
            answer            = _format_member_answer(result),
            route             = route,
            confidence        = route_info["confidence"],
            sources           = [],
            model             = "sql",
            prompt_tokens     = 0,
            completion_tokens = 0,
            latency_ms        = _ms(start_time),
            member_id         = member_id,
            claim_id          = None,
        )

    if route == QueryRoute.SQL_AUTH and route_info.get("auth_id"):
        result = lookup_auth(route_info["auth_id"])
        if not result["found"]:
            raise HTTPException(404, detail=f"Auth {route_info['auth_id']} not found")
        return QueryResponse(
            answer            = _format_auth_answer(result),
            route             = route,
            confidence        = route_info["confidence"],
            sources           = [],
            model             = "sql",
            prompt_tokens     = 0,
            completion_tokens = 0,
            latency_ms        = _ms(start_time),
            member_id         = result.get("member_id"),
            claim_id          = None,
        )

    chunks = vector_search(
        question  = request.question,
        top_k     = request.top_k,
        member_id = member_id,
        doc_type  = request.doc_type,
        claim_id  = claim_id,
    )

    if not chunks:
        return QueryResponse(
            answer            = "No relevant documents found for this query.",
            route             = route,
            confidence        = "low",
            sources           = [],
            model             = "none",
            prompt_tokens     = 0,
            completion_tokens = 0,
            latency_ms        = _ms(start_time),
            member_id         = member_id,
            claim_id          = claim_id,
        )

    messages   = build_prompt(
        question         = request.question,
        retrieved_chunks = chunks,
        route_info       = route_info,
    )
    llm_result = generate(messages)
    latency_ms = _ms(start_time)

    _log_query_to_mlflow(
        question   = request.question,
        route      = route,
        chunks     = chunks,
        llm_result = llm_result,
        latency_ms = latency_ms,
    )

    return QueryResponse(
        answer            = llm_result["answer"],
        route             = route,
        confidence        = route_info["confidence"],
        sources           = [SourceDocument(**s) for s in format_sources(chunks)],
        model             = llm_result["model"],
        prompt_tokens     = llm_result["prompt_tokens"],
        completion_tokens = llm_result["completion_tokens"],
        latency_ms        = latency_ms,
        member_id         = member_id,
        claim_id          = claim_id,
    )


@app.get("/claim/{claim_id}", response_model=ClaimResponse)
async def get_claim(claim_id: str):
    result = lookup_claim(claim_id.upper())
    if not result["found"]:
        raise HTTPException(404, detail=f"Claim {claim_id} not found")
    return ClaimResponse(**result)


@app.get("/member/{member_id}", response_model=MemberResponse)
async def get_member(member_id: str):
    result = lookup_member(member_id.upper())
    if not result["found"]:
        raise HTTPException(404, detail=f"Member {member_id} not found")
    return MemberResponse(**result)


@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        stats        = get_pinecone_index().describe_index_stats()
        vector_count = stats.total_vector_count
    except Exception:
        vector_count = -1

    try:
        from sqlalchemy import text
        with get_db_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    return HealthResponse(
        status           = "healthy" if db_ok and vector_count > 0 else "degraded",
        pinecone_vectors = vector_count,
        db_connected     = db_ok,
        embedding_model  = "all-MiniLM-L6-v2",
        llm_provider     = LLM_PROVIDER,
        environment      = ENVIRONMENT,
    )


@app.get("/metrics")
async def metrics():
    from sqlalchemy import text
    with get_db_engine().connect() as conn:
        members = conn.execute(text("SELECT COUNT(*) FROM members")).scalar()
        claims  = conn.execute(text("SELECT COUNT(*) FROM claims")).scalar()
        denied  = conn.execute(
            text("SELECT COUNT(*) FROM claims WHERE status='denied'")
        ).scalar()
        auths = conn.execute(text("SELECT COUNT(*) FROM prior_auth")).scalar()

    stats = get_pinecone_index().describe_index_stats()

    return {
        "database": {
            "members":       members,
            "claims":        claims,
            "denied_claims": denied,
            "prior_auths":   auths,
        },
        "vector_store": {
            "total_vectors":   stats.total_vector_count,
            "index_name":      os.getenv("PINECONE_INDEX_NAME"),
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dims":  384,
        },
        "api": {
            "environment": ENVIRONMENT,
            "llm_provider": LLM_PROVIDER,
        },
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    start     = time.time()
    doc_types = [request.doc_type.value] if request.doc_type else None
    _run_ingestion(doc_types, request.limit)
    return IngestResponse(
        status           = "completed",
        files_processed  = 0,
        chunks_created   = 0,
        vectors_upserted = 0,
        errors           = 0,
        duration_seconds = round(time.time() - start, 2),
    )


def _ms(start: float) -> float:
    return round((time.time() - start) * 1000, 2)


def _format_claim_answer(r: dict) -> str:
    lines = [
        f"Claim {r['claim_id']} — {r['status'].upper()}",
        f"Member: {r['member_name']} ({r['member_id']})",
        f"Plan: {r['plan_type']}",
        f"Procedure: CPT {r['cpt_code']}",
        f"Diagnosis: ICD-10 {r['icd_code']}",
        f"Billed: ${r['billed_amount']:,.2f} | "
        f"Allowed: ${r['allowed_amount']:,.2f} | "
        f"Plan Paid: ${r['plan_paid']:,.2f}",
        f"Service Date: {r['service_date']}",
        f"Processed: {r['processed_date']}",
    ]
    if r.get("denial_code"):
        lines.append(f"Denial Code: {r['denial_code']}")
    lines.append("Source: claims database (structured lookup)")
    return "\n".join(lines)


def _format_member_answer(r: dict) -> str:
    s = r["claim_summary"]
    return (
        f"Member {r['member_id']} — {r['status'].upper()}\n"
        f"Name: {r['name']}\n"
        f"Plan: {r['plan_type']} | State: {r['state']}\n"
        f"Enrolled: {r['enrollment_date']}\n"
        f"Claims: {s['total']} total | "
        f"{s['approved']} approved | {s['denied']} denied | {s['pending']} pending\n"
        f"Total Billed: ${s['total_billed']:,.2f} | "
        f"Total Paid: ${s['total_paid']:,.2f}\n"
        f"Source: members database (structured lookup)"
    )


def _format_auth_answer(r: dict) -> str:
    return (
        f"Authorization {r['auth_id']} — {r['decision'].upper()}\n"
        f"Member: {r['member_name']} ({r['member_id']})\n"
        f"Procedure: CPT {r['procedure_code']}\n"
        f"Requested: {r['request_date']} | Decision: {r['decision_date']}\n"
        f"Source: prior_auth database (structured lookup)"
    )


def _log_query_to_mlflow(question, route, chunks, llm_result, latency_ms):
    try:
        with mlflow.start_run(nested=True):
            mlflow.log_param("question_length", len(question))
            mlflow.log_param("route",           str(route))
            mlflow.log_param("model",           llm_result["model"])
            mlflow.log_metric("chunks_retrieved",  len(chunks))
            mlflow.log_metric("top_score",         chunks[0]["score"] if chunks else 0)
            mlflow.log_metric("prompt_tokens",     llm_result["prompt_tokens"])
            mlflow.log_metric("completion_tokens", llm_result["completion_tokens"])
            mlflow.log_metric("latency_ms",        latency_ms)
    except Exception:
        pass


def _run_ingestion(doc_types, limit):
    from ingestion.ingest import ingest
    ingest(doc_types=doc_types, limit=limit)
