"""
ingestion/ingest.py

Document ingestion pipeline for the UHG RAG system.

Reads all document types from data/raw/, extracts text based on format,
chunks with LangChain, embeds with sentence-transformers, and upserts
to Pinecone with metadata. Designed to run nightly via Airflow.

Run:
    python ingestion/ingest.py
    python ingestion/ingest.py --doc-type eob --limit 10
"""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import fitz
import mlflow
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 50
BATCH_SIZE      = 100

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX_NAME", "uhg-claims-rag")
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

DOC_FOLDERS = {
    "eob":           RAW_DIR / "eob",
    "clinical_note": RAW_DIR / "clinical_notes",
    "prior_auth":    RAW_DIR / "prior_auth",
    "denial":        RAW_DIR / "denial_letters",
    "appeal":        RAW_DIR / "appeal_letters",
    "fhir":          RAW_DIR / "fhir",
    "case_note":     RAW_DIR / "case_notes",
}

PROGRESS_FILE = PROCESSED_DIR / "ingested_files.json"


def extract_pdf(path: Path) -> str:
    doc  = fitz.open(str(path))
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()


def extract_json_fhir(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        record = json.load(f)

    parts = []
    for entry in record.get("entry", []):
        rt = entry.get("resourceType", "")

        if rt == "Patient":
            parts.append(
                f"Patient: {entry.get('name')} | "
                f"Member ID: {entry.get('id')} | "
                f"DOB: {entry.get('birthDate')} | "
                f"Plan: {entry.get('insurance', {}).get('plan')} | "
                f"Status: {entry.get('insurance', {}).get('status')}"
            )
        elif rt == "ConditionList":
            for c in entry.get("conditions", []):
                parts.append(f"Active condition: {c['display']} (ICD-10: {c['code']})")
        elif rt == "MedicationList":
            meds = ", ".join(entry.get("medications", []))
            parts.append(f"Current medications: {meds}")
        elif rt == "AllergyList":
            allergies = ", ".join(entry.get("allergies", []))
            parts.append(f"Known allergies: {allergies}")
        elif rt == "EncounterList":
            for e in entry.get("encounters", []):
                parts.append(
                    f"Encounter on {e['date']}: {e['type']} "
                    f"(CPT: {e['cpt_code']}) — Status: {e['status']} "
                    f"— Provider: {e['provider']}"
                )
        elif rt == "LabResults":
            for r in entry.get("results", []):
                parts.append(f"Lab result — {r['test']}: {r['value']} {r['unit']}")

    return "\n".join(parts)


def extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def extract_text(path: Path, doc_type: str) -> str:
    if doc_type == "fhir":
        return extract_json_fhir(path)
    elif doc_type == "case_note":
        return extract_txt(path)
    else:
        return extract_pdf(path)


def parse_metadata(path: Path, doc_type: str) -> dict:
    stem  = path.stem
    parts = stem.split("_")
    meta  = {
        "doc_type":    doc_type,
        "source_file": path.name,
        "ingested_at": datetime.now().isoformat(),
    }
    try:
        if doc_type in ("eob", "clinical_note", "denial", "appeal"):
            meta["claim_id"]  = parts[1]
            meta["member_id"] = parts[2]
            meta["date"]      = parts[3]
        elif doc_type == "prior_auth":
            meta["auth_id"]   = parts[1]
            meta["member_id"] = parts[2]
            meta["date"]      = parts[3]
        elif doc_type == "fhir":
            meta["member_id"] = parts[1]
        elif doc_type == "case_note":
            meta["claim_id"]  = parts[1]
            meta["member_id"] = parts[2]
    except (IndexError, KeyError):
        pass
    return meta


def make_chunk_id(source_file: str, chunk_index: int) -> str:
    base = f"{source_file}__chunk_{chunk_index}"
    return hashlib.md5(base.encode()).hexdigest()


def load_progress() -> set:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()


def save_progress(ingested: set):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(ingested), f)


def ingest(doc_types: list = None, limit: int = None):
    if doc_types is None:
        doc_types = list(DOC_FOLDERS.keys())

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Model loaded. Output dim: {model.get_sentence_embedding_dimension()}")

    print(f"\nConnecting to Pinecone index: {PINECONE_INDEX}")
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    stats = index.describe_index_stats()
    print(f"  Index stats: {stats.total_vector_count:,} vectors currently stored")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", " ", ""],
    )

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("uhg-rag-ingestion")

    ingested_files = load_progress()
    total_stats = {
        "files_processed":  0,
        "files_skipped":    0,
        "chunks_created":   0,
        "vectors_upserted": 0,
        "errors":           0,
    }

    with mlflow.start_run(run_name=f"ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("embedding_model", EMBEDDING_MODEL)
        mlflow.log_param("chunk_size",      CHUNK_SIZE)
        mlflow.log_param("chunk_overlap",   CHUNK_OVERLAP)
        mlflow.log_param("doc_types",       str(doc_types))

        for doc_type in doc_types:
            folder = DOC_FOLDERS.get(doc_type)
            if not folder or not folder.exists():
                print(f"\n  Skipping {doc_type} — folder not found")
                continue

            files = sorted(folder.iterdir())
            if limit:
                files = files[:limit]

            print(f"\n{'=' * 50}")
            print(f"Processing: {doc_type.upper()} ({len(files)} files)")
            print(f"{'=' * 50}")

            batch_vectors = []

            for i, file_path in enumerate(files):
                if file_path.name in ingested_files:
                    total_stats["files_skipped"] += 1
                    continue

                try:
                    text = extract_text(file_path, doc_type)
                    if not text or len(text) < 50:
                        continue

                    metadata = parse_metadata(file_path, doc_type)
                    chunks   = splitter.split_text(text)

                    embeddings = model.encode(
                        chunks,
                        batch_size=32,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    )

                    for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        chunk_meta = {
                            **metadata,
                            "chunk_index": chunk_idx,
                            "chunk_text":  chunk[:500],
                        }
                        batch_vectors.append({
                            "id":       make_chunk_id(file_path.name, chunk_idx),
                            "values":   embedding.tolist(),
                            "metadata": chunk_meta,
                        })
                        total_stats["chunks_created"] += 1

                    if len(batch_vectors) >= BATCH_SIZE:
                        index.upsert(vectors=batch_vectors)
                        total_stats["vectors_upserted"] += len(batch_vectors)
                        batch_vectors = []

                    ingested_files.add(file_path.name)
                    total_stats["files_processed"] += 1

                    if (i + 1) % 100 == 0:
                        print(f"  [{i + 1}/{len(files)}] {doc_type} — "
                              f"{total_stats['chunks_created']:,} chunks so far")

                except Exception as e:
                    print(f"  ERROR on {file_path.name}: {e}")
                    total_stats["errors"] += 1
                    continue

            if batch_vectors:
                index.upsert(vectors=batch_vectors)
                total_stats["vectors_upserted"] += len(batch_vectors)
                batch_vectors = []

            print(f"  Done: {doc_type} — processed so far: {total_stats['files_processed']}")

        save_progress(ingested_files)

        for k, v in total_stats.items():
            mlflow.log_metric(k, v)

        time.sleep(2)
        final_stats = index.describe_index_stats()
        mlflow.log_metric("pinecone_total_vectors", final_stats.total_vector_count)

    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)
    for k, v in total_stats.items():
        print(f"  {k}: {v:,}")
    print(f"  Pinecone total vectors: {final_stats.total_vector_count:,}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UHG RAG ingestion pipeline")
    parser.add_argument(
        "--doc-type",
        choices=list(DOC_FOLDERS.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    doc_types = list(DOC_FOLDERS.keys()) if args.doc_type == "all" else [args.doc_type]
    ingest(doc_types=doc_types, limit=args.limit)
