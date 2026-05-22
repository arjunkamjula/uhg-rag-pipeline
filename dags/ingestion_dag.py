"""
dags/ingestion_dag.py

Airflow DAG — UHG RAG nightly ingestion pipeline

Schedule: 2am every night (0 2 * * *)
Tasks:
  1. check_new_documents  — detect files not yet in Pinecone
  2. run_ingestion        — extract, chunk, embed, upsert to Pinecone
  3. validate_vectors     — assert Pinecone vector count is non-zero
  4. log_run_summary      — write summary metrics to MLflow

ShortCircuitOperator on task 1 skips the rest if no new documents are found,
avoiding unnecessary compute and LLM/embedding costs.

XCom is used to pass values between tasks without a shared database.
"""

import json
import os
from datetime import timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago

DEFAULT_ARGS = {
    "owner":            "uhg-ai-team",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

RAW_DIR       = Path(os.getenv("RAW_DIR", "data/raw"))
PROGRESS_FILE = Path("data/processed/ingested_files.json")


def check_new_documents(**context):
    """
    Scan raw directory for files not yet in the progress tracker.
    Returns True to continue, False to short-circuit downstream tasks.
    Pushes new_file_count to XCom so log_run_summary can read it.
    """
    already_ingested = set()
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            already_ingested = set(json.load(f))

    new_files = []
    for folder in RAW_DIR.iterdir():
        if folder.is_dir():
            for file in folder.iterdir():
                if file.name not in already_ingested:
                    new_files.append(str(file))

    print(f"Found {len(new_files)} new files to ingest")
    context["ti"].xcom_push(key="new_file_count", value=len(new_files))
    return len(new_files) > 0


def run_ingestion(**context):
    """
    Import and call the ingestion pipeline directly.
    Runs inside the Airflow worker process — same as running
    python ingestion/ingest.py manually, but scheduled.
    """
    import sys
    sys.path.insert(0, os.getenv("PIPELINE_ROOT", "/opt/airflow"))

    from ingestion.ingest import ingest

    print("Starting ingestion pipeline...")
    ingest(doc_types=None, limit=None)
    print("Ingestion pipeline complete.")


def validate_vectors(**context):
    """
    Query Pinecone after ingestion to confirm vectors were written.
    Asserts count > 0 — if ingestion silently failed, this task fails
    and Airflow alerts the team before the next query run.
    """
    import os
    from pinecone import Pinecone
    from dotenv import load_dotenv

    load_dotenv()

    pc    = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "uhg-claims-rag"))
    stats = index.describe_index_stats()
    count = stats.total_vector_count

    print(f"Pinecone vector count after ingestion: {count:,}")
    context["ti"].xcom_push(key="vector_count", value=count)

    assert count > 0, f"Vector count is {count} — ingestion may have failed silently"
    print("Vector count validation passed.")


def log_run_summary(**context):
    """
    Pull XCom values from upstream tasks and write a summary run to MLflow.
    This appears in the MLflow UI under experiment 'uhg-rag-ingestion'.
    """
    import mlflow
    from dotenv import load_dotenv

    load_dotenv()

    ti             = context["ti"]
    new_file_count = ti.xcom_pull(task_ids="check_new_documents", key="new_file_count") or 0
    vector_count   = ti.xcom_pull(task_ids="validate_vectors",    key="vector_count")   or 0
    run_date       = context["ds"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("uhg-rag-ingestion")

    with mlflow.start_run(run_name=f"airflow_nightly_{run_date.replace('-', '')}"):
        mlflow.log_param("trigger",   "airflow_schedule")
        mlflow.log_param("run_date",   run_date)
        mlflow.log_metric("new_files",     new_file_count)
        mlflow.log_metric("total_vectors", vector_count)

    print(f"MLflow run logged — date={run_date} new_files={new_file_count} vectors={vector_count}")


with DAG(
    dag_id            = "uhg_rag_nightly_ingestion",
    default_args      = DEFAULT_ARGS,
    description       = "Nightly ingestion of clinical and claims documents into Pinecone",
    schedule_interval = "0 2 * * *",
    start_date        = days_ago(1),
    catchup           = False,
    tags              = ["uhg", "rag", "ingestion"],
) as dag:

    t1_check = ShortCircuitOperator(
        task_id         = "check_new_documents",
        python_callable = check_new_documents,
        provide_context = True,
    )

    t2_ingest = PythonOperator(
        task_id         = "run_ingestion",
        python_callable = run_ingestion,
        provide_context = True,
    )

    t3_validate = PythonOperator(
        task_id         = "validate_vectors",
        python_callable = validate_vectors,
        provide_context = True,
    )

    t4_log = PythonOperator(
        task_id         = "log_run_summary",
        python_callable = log_run_summary,
        provide_context = True,
    )

    t1_check >> t2_ingest >> t3_validate >> t4_log
