"""
api/dependencies.py

Shared singletons loaded once at application startup via FastAPI lifespan.
lru_cache ensures each object is created exactly once and reused across requests.
"""

import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]      = "1"

from functools import lru_cache
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from sqlalchemy import create_engine

load_dotenv()

EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX_NAME", "uhg-claims-rag")
DB_URL           = os.getenv(
    "DATABASE_URL",
    "postgresql://uhg_user:uhg_pass@localhost:5433/uhg_claims"
)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX)


@lru_cache(maxsize=1)
def get_db_engine():
    return create_engine(DB_URL, pool_pre_ping=True, pool_size=5)
