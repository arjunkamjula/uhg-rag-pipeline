# UHG RAG Pipeline

A document intelligence pipeline that lets internal users and downstream systems query unstructured clinical and claims documents in natural language. Built to reduce manual document lookup time for claims processors working across EOBs, clinical notes, prior authorization letters, denial letters, and FHIR patient records.

## What it does

Claims processors previously spent 15-20 minutes per claim manually reading through multiple documents to answer questions like "was this procedure authorized?" or "what was the denial reason?". This pipeline ingests all those documents into a vector store and exposes a query API that routes each question to either a direct SQL lookup (for structured queries with exact IDs) or a RAG pipeline (for reasoning queries that require reading document content).

## Architecture

```
Data sources (PDF, FHIR JSON, TXT, CSV)
        |
        | nightly batch via Airflow
        v
Ingestion pipeline (PyMuPDF -> LangChain splitter -> sentence-transformers -> Pinecone)
        |
        v
Storage: Pinecone (vectors) + PostgreSQL (structured claims data)
        |
        v
Query pipeline: router -> SQL lookup OR vector search -> prompt builder -> LLM
        |
        v
FastAPI service (POST /query, GET /claim, GET /member, GET /health, GET /metrics)
        |
        v
MLflow (query logs, latency, token cost, RAGAS evaluation scores)
```

## Stack

- **Ingestion**: PyMuPDF, LangChain RecursiveCharacterTextSplitter, sentence-transformers (all-MiniLM-L6-v2)
- **Vector store**: Pinecone (384-dim, cosine similarity, metadata filtering)
- **Structured DB**: PostgreSQL
- **LLM**: Groq (Llama-3.3-70b) / OpenAI GPT-4o (one env var swap)
- **API**: FastAPI + Uvicorn + Pydantic
- **Orchestration**: Apache Airflow (nightly DAG)
- **Monitoring**: MLflow (experiment tracking, query logs, evaluation scores)
- **Evaluation**: RAGAS (faithfulness, answer relevancy, context precision, context recall)
- **Deployment**: Docker, Kubernetes, GitHub Actions CI/CD

## Setup

### 1. Clone and configure

```bash
git clone https://github.com/your-username/uhg-rag-pipeline.git
cd uhg-rag-pipeline
cp .env.example .env
```

Edit `.env` with your API keys:
```
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=uhg-claims-rag
GROQ_API_KEY=your_key
DATABASE_URL=postgresql://uhg_user:uhg_pass@localhost:5433/uhg_claims
```

### 2. Create Pinecone index

Go to app.pinecone.io and create an index:
- Name: `uhg-claims-rag`
- Dimensions: `384`
- Metric: `cosine`
- Serverless, AWS us-east-1

### 3. Start infrastructure

```bash
docker-compose up -d
```

Services:
- PostgreSQL on port 5433
- MLflow on port 5000 (`http://localhost:5000`)
- Airflow on port 8080 (`http://localhost:8080`, admin/admin)

### 4. Install dependencies

```bash
conda create -n uhg-rag python=3.11 -y
conda activate uhg-rag
pip install -r requirements.txt
pip install -e .
```

### 5. Generate data and ingest

```bash
python ingestion/generate_data.py
python ingestion/ingest.py
```

### 6. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

API docs at `http://localhost:8000/docs`

## API

### POST /query

Main query endpoint. Routes automatically to SQL or RAG.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Why was claim CLM-10001 denied?", "top_k": 4}'
```

```bash
# with member filter — uses focused RAG
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What procedures has this member had?", "member_id": "M-19123"}'
```

### GET /claim/{claim_id}

Direct SQL lookup for a claim. No LLM, ~6ms.

```bash
curl http://localhost:8000/claim/CLM-10001
```

### GET /member/{member_id}

Member eligibility and claim summary.

```bash
curl http://localhost:8000/member/M-19123
```

### GET /health

```bash
curl http://localhost:8000/health
```

## Routing logic

Every query hits the router first:

| Query pattern | Route | Cost | Latency |
|---|---|---|---|
| Exact claim ID, no reasoning keyword | SQL | $0 | ~6ms |
| Exact member ID + simple field | SQL | $0 | ~6ms |
| ID + reasoning keyword (why, explain, denied) | RAG focused | ~$0.001 | ~2300ms |
| Reasoning keyword, no ID | RAG vague | ~$0.001 | ~2300ms |
| member_id in request body | RAG focused | ~$0.001 | ~2300ms |

About 40% of queries route to SQL at zero LLM cost.

## Evaluation

```bash
python evaluation/evaluate.py
python evaluation/evaluate.py --sample 5
```

Scores logged to MLflow under experiment `uhg-rag-evaluation`.

RAGAS metrics:
- **faithfulness** — is every claim in the answer grounded in retrieved chunks?
- **answer_relevancy** — does the answer address the question?
- **context_precision** — are retrieved chunks relevant?
- **context_recall** — were all needed chunks retrieved?

## Running tests

```bash
pytest tests/ -v
```

## Deployment

Docker:
```bash
docker build -t uhg-rag-pipeline .
docker run -p 8000:8000 --env-file .env uhg-rag-pipeline
```

Kubernetes:
```bash
kubectl create namespace uhg-ai
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```
