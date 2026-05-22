"""
retrieval/sql_lookup.py

Direct PostgreSQL lookups for structured queries.
No embedding, no LLM — just SQL.
Handles ~40% of queries at near-zero cost and ~6ms latency.
"""

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://uhg_user:uhg_pass@localhost:5433/uhg_claims"
)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, pool_pre_ping=True)
    return _engine


def lookup_claim(claim_id: str) -> dict:
    sql = text("""
        SELECT
            c.claim_id,
            c.member_id,
            m.name          AS member_name,
            m.plan_type,
            c.cpt_code,
            c.icd_code,
            c.billed_amount,
            c.allowed_amount,
            c.plan_paid,
            c.status,
            c.service_date::text,
            c.processed_date::text,
            c.provider_npi,
            c.denial_code
        FROM claims c
        JOIN members m ON c.member_id = m.member_id
        WHERE c.claim_id = :claim_id
    """)

    with get_engine().connect() as conn:
        row = conn.execute(sql, {"claim_id": claim_id}).fetchone()

    if not row:
        return {"found": False, "claim_id": claim_id}

    return {
        "found":          True,
        "source":         "sql",
        "claim_id":       row.claim_id,
        "member_id":      row.member_id,
        "member_name":    row.member_name,
        "plan_type":      row.plan_type,
        "cpt_code":       row.cpt_code,
        "icd_code":       row.icd_code,
        "billed_amount":  float(row.billed_amount),
        "allowed_amount": float(row.allowed_amount),
        "plan_paid":      float(row.plan_paid),
        "status":         row.status,
        "service_date":   row.service_date,
        "processed_date": row.processed_date,
        "provider_npi":   row.provider_npi,
        "denial_code":    row.denial_code,
    }


def lookup_member(member_id: str) -> dict:
    member_sql = text("""
        SELECT
            member_id, name, dob::text, plan_type,
            state, enrollment_date::text, status
        FROM members
        WHERE member_id = :member_id
    """)

    summary_sql = text("""
        SELECT
            COUNT(*)                                            AS total_claims,
            SUM(CASE WHEN status='approved' THEN 1 ELSE 0 END) AS approved,
            SUM(CASE WHEN status='denied'   THEN 1 ELSE 0 END) AS denied,
            SUM(CASE WHEN status='pending'  THEN 1 ELSE 0 END) AS pending,
            ROUND(SUM(billed_amount)::numeric, 2)              AS total_billed,
            ROUND(SUM(plan_paid)::numeric,     2)              AS total_paid
        FROM claims
        WHERE member_id = :member_id
    """)

    with get_engine().connect() as conn:
        member_row  = conn.execute(member_sql,  {"member_id": member_id}).fetchone()
        summary_row = conn.execute(summary_sql, {"member_id": member_id}).fetchone()

    if not member_row:
        return {"found": False, "member_id": member_id}

    return {
        "found":           True,
        "source":          "sql",
        "member_id":       member_row.member_id,
        "name":            member_row.name,
        "dob":             member_row.dob,
        "plan_type":       member_row.plan_type,
        "state":           member_row.state,
        "enrollment_date": member_row.enrollment_date,
        "status":          member_row.status,
        "claim_summary": {
            "total":       int(summary_row.total_claims or 0),
            "approved":    int(summary_row.approved     or 0),
            "denied":      int(summary_row.denied       or 0),
            "pending":     int(summary_row.pending      or 0),
            "total_billed": float(summary_row.total_billed or 0),
            "total_paid":   float(summary_row.total_paid   or 0),
        },
    }


def lookup_auth(auth_id: str) -> dict:
    sql = text("""
        SELECT
            a.auth_id,
            a.member_id,
            m.name       AS member_name,
            m.plan_type,
            a.procedure_code,
            a.decision,
            a.request_date::text,
            a.decision_date::text
        FROM prior_auth a
        JOIN members m ON a.member_id = m.member_id
        WHERE a.auth_id = :auth_id
    """)

    with get_engine().connect() as conn:
        row = conn.execute(sql, {"auth_id": auth_id}).fetchone()

    if not row:
        return {"found": False, "auth_id": auth_id}

    return {
        "found":          True,
        "source":         "sql",
        "auth_id":        row.auth_id,
        "member_id":      row.member_id,
        "member_name":    row.member_name,
        "plan_type":      row.plan_type,
        "procedure_code": row.procedure_code,
        "decision":       row.decision,
        "request_date":   row.request_date,
        "decision_date":  row.decision_date,
    }
