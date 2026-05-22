"""
api/schemas.py

Pydantic request and response models for all API endpoints.
FastAPI validates every incoming request against these schemas automatically.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class DocType(str, Enum):
    eob           = "eob"
    clinical_note = "clinical_note"
    prior_auth    = "prior_auth"
    denial        = "denial"
    appeal        = "appeal"
    fhir          = "fhir"
    case_note     = "case_note"


class IngestRequest(BaseModel):
    doc_type: Optional[DocType] = Field(None, description="Ingest specific doc type only")
    limit:    Optional[int]     = Field(None, description="Max files per doc type")


class IngestResponse(BaseModel):
    status:           str
    files_processed:  int
    chunks_created:   int
    vectors_upserted: int
    errors:           int
    duration_seconds: float


class QueryRequest(BaseModel):
    question:  str              = Field(..., min_length=3)
    member_id: Optional[str]   = Field(None)
    doc_type:  Optional[DocType] = Field(None)
    top_k:     int              = Field(4, ge=1, le=10)


class SourceDocument(BaseModel):
    source_file:  str
    doc_type:     str
    member_id:    Optional[str]
    claim_id:     Optional[str]
    date:         Optional[str]
    score:        float
    chunk_index:  int


class QueryResponse(BaseModel):
    answer:            str
    route:             str
    confidence:        str
    sources:           list[SourceDocument]
    model:             str
    prompt_tokens:     int
    completion_tokens: int
    latency_ms:        float
    member_id:         Optional[str]
    claim_id:          Optional[str]


class ClaimResponse(BaseModel):
    found:          bool
    source:         Optional[str]
    claim_id:       Optional[str]
    member_id:      Optional[str]
    member_name:    Optional[str]
    plan_type:      Optional[str]
    cpt_code:       Optional[str]
    icd_code:       Optional[str]
    billed_amount:  Optional[float]
    allowed_amount: Optional[float]
    plan_paid:      Optional[float]
    status:         Optional[str]
    service_date:   Optional[str]
    processed_date: Optional[str]
    provider_npi:   Optional[str]
    denial_code:    Optional[str]


class MemberResponse(BaseModel):
    found:           bool
    source:          Optional[str]
    member_id:       Optional[str]
    name:            Optional[str]
    dob:             Optional[str]
    plan_type:       Optional[str]
    state:           Optional[str]
    enrollment_date: Optional[str]
    status:          Optional[str]
    claim_summary:   Optional[dict]


class HealthResponse(BaseModel):
    status:           str
    pinecone_vectors: int
    db_connected:     bool
    embedding_model:  str
    llm_provider:     str
    environment:      str
