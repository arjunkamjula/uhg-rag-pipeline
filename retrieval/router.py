"""
retrieval/router.py

Query intent router — every request hits this before any search or LLM call.

Routes to:
  SQL_CLAIM   — exact claim ID lookup against PostgreSQL
  SQL_MEMBER  — exact member ID + simple field lookup
  SQL_AUTH    — exact auth ID lookup
  RAG_FOCUSED — semantic search with metadata filter (member/claim ID present)
  RAG_VAGUE   — wide semantic search with no filter

About 40% of production queries route to SQL at zero LLM cost.
"""

import re
from enum import Enum


class QueryRoute(str, Enum):
    SQL_CLAIM   = "sql_claim"
    SQL_MEMBER  = "sql_member"
    SQL_AUTH    = "sql_auth"
    RAG_FOCUSED = "rag_focused"
    RAG_VAGUE   = "rag_vague"


RE_CLAIM_ID  = re.compile(r'\bCLM[-]?\d{4,6}\b', re.IGNORECASE)
RE_MEMBER_ID = re.compile(r'\bM[-]?\d{4,6}\b',   re.IGNORECASE)
RE_AUTH_ID   = re.compile(r'\bPA[-]?\d{3,5}\b',   re.IGNORECASE)

REASONING_KEYWORDS = [
    "why", "explain", "reason", "justif",
    "summarize", "summary", "describe",
    "should", "recommend", "suggest",
    "history", "background", "tell me about",
    "what happened", "what was", "what is",
    "medically necessary", "medical necessity",
    "criteria", "eligible", "qualify",
    "denied", "denial", "appeal", "overturn",
    "approved", "authorization", "prior auth",
    "clinical", "diagnosis", "treatment", "procedure",
]

SIMPLE_FIELD_KEYWORDS = [
    "status", "active", "is active", "is eligible", "eligible",
    "plan type", "plan", "enrollment", "member since",
    "how much", "amount", "balance", "paid",
    "count", "how many", "list all",
]


def detect_route(question: str) -> dict:
    q       = question.strip()
    q_lower = q.lower()

    claim_match  = RE_CLAIM_ID.search(q)
    member_match = RE_MEMBER_ID.search(q)
    auth_match   = RE_AUTH_ID.search(q)

    claim_id  = _normalise_claim_id(claim_match.group())   if claim_match  else None
    member_id = _normalise_member_id(member_match.group()) if member_match else None
    auth_id   = _normalise_auth_id(auth_match.group())     if auth_match   else None

    if claim_id and not _has_reasoning(q_lower):
        return _result(
            QueryRoute.SQL_CLAIM, claim_id, member_id, auth_id,
            confidence="high",
            reason=f"Exact claim ID {claim_id} detected. SQL lookup.",
        )

    if auth_id and not _has_reasoning(q_lower):
        return _result(
            QueryRoute.SQL_AUTH, claim_id, member_id, auth_id,
            confidence="high",
            reason=f"Auth ID {auth_id} detected. SQL lookup on prior_auth table.",
        )

    if member_id and _has_simple_field(q_lower) and not _has_reasoning(q_lower):
        return _result(
            QueryRoute.SQL_MEMBER, claim_id, member_id, auth_id,
            confidence="high",
            reason=f"Member ID {member_id} with simple field query. SQL lookup.",
        )

    if (claim_id or member_id) and _has_reasoning(q_lower):
        return _result(
            QueryRoute.RAG_FOCUSED, claim_id, member_id, auth_id,
            confidence="high",
            reason=f"ID detected ({claim_id or member_id}) with reasoning keyword. RAG with metadata filter.",
        )

    if _has_reasoning(q_lower):
        return _result(
            QueryRoute.RAG_VAGUE, claim_id, member_id, auth_id,
            confidence="medium",
            reason="Reasoning keyword detected but no specific ID. Wide RAG search.",
        )

    return _result(
        QueryRoute.RAG_VAGUE, claim_id, member_id, auth_id,
        confidence="low",
        reason="No identifiers or reasoning keywords detected. Falling back to wide RAG search.",
    )


def _has_reasoning(q_lower: str) -> bool:
    return any(kw in q_lower for kw in REASONING_KEYWORDS)


def _has_simple_field(q_lower: str) -> bool:
    return any(kw in q_lower for kw in SIMPLE_FIELD_KEYWORDS)


def _normalise_claim_id(raw: str) -> str:
    digits = re.sub(r'\D', '', raw)
    return f"CLM-{digits}"


def _normalise_member_id(raw: str) -> str:
    digits = re.sub(r'\D', '', raw)
    return f"M-{digits}"


def _normalise_auth_id(raw: str) -> str:
    digits = re.sub(r'\D', '', raw)
    return f"PA-{digits}"


def _result(route, claim_id, member_id, auth_id, confidence, reason):
    return {
        "route":      route,
        "claim_id":   claim_id,
        "member_id":  member_id,
        "auth_id":    auth_id,
        "confidence": confidence,
        "reason":     reason,
    }
