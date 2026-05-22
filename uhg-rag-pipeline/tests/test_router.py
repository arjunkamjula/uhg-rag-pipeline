"""
tests/test_router.py

Unit tests for the query intent router.
Tests that routing decisions are correct for each query type.
"""

import pytest
from retrieval.router import detect_route, QueryRoute


def test_exact_claim_id_routes_to_sql():
    result = detect_route("Show me claim CLM-10001")
    assert result["route"] == QueryRoute.SQL_CLAIM
    assert result["claim_id"] == "CLM-10001"


def test_claim_id_with_reasoning_routes_to_rag():
    result = detect_route("Why was claim CLM-10001 denied?")
    assert result["route"] == QueryRoute.RAG_FOCUSED
    assert result["claim_id"] == "CLM-10001"


def test_member_id_active_routes_to_sql():
    result = detect_route("Is member M-19123 active?")
    assert result["member_id"] == "M-19123"


def test_auth_id_routes_to_sql():
    result = detect_route("Show me auth PA-1234")
    assert result["route"] == QueryRoute.SQL_AUTH
    assert result["auth_id"] == "PA-1234"


def test_vague_reasoning_routes_to_rag_vague():
    result = detect_route("What is the medical justification for knee replacement?")
    assert result["route"] == QueryRoute.RAG_VAGUE


def test_no_id_no_keyword_routes_to_rag_vague():
    result = detect_route("show me something")
    assert result["route"] == QueryRoute.RAG_VAGUE
    assert result["confidence"] == "low"


def test_member_id_extracted_correctly():
    result = detect_route("Why was claim CLM-99999 denied for member M-55555?")
    assert result["claim_id"] == "CLM-99999"
    assert result["member_id"] == "M-55555"
    assert result["route"] == QueryRoute.RAG_FOCUSED
