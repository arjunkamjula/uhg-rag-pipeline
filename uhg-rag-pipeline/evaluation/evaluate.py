"""
evaluation/evaluate.py

RAGAS evaluation harness for the UHG RAG pipeline.

Runs a set of test questions with known expected answers through the
retrieval + generation pipeline and scores them on four metrics:
  - faithfulness       : does the LLM answer stay within the retrieved chunks?
  - answer_relevancy   : does the answer actually address the question?
  - context_precision  : are the retrieved chunks relevant to the question?
  - context_recall     : did we retrieve all chunks needed to answer?

Results are logged to MLflow under experiment 'uhg-rag-evaluation'.

Run:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --sample 10
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import mlflow
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from retrieval.vector_search import search as vector_search
from retrieval.prompt_builder import build_prompt
from retrieval.llm_client import generate

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EVAL_DIR   = Path("evaluation")

# hand-labeled test set
# each entry has a question, the expected answer, and the member/claim context
TEST_QUESTIONS = [
    {
        "question":       "Was CPT-27447 approved for member M-19123?",
        "member_id":      "M-19123",
        "ground_truth":   "The claim for CPT-27447 total knee arthroplasty was approved.",
    },
    {
        "question":       "What is the denial reason for a claim with denial code CO-50?",
        "member_id":      None,
        "ground_truth":   "CO-50 means the service was not medically necessary.",
    },
    {
        "question":       "What medications is the patient currently taking?",
        "member_id":      "M-19123",
        "ground_truth":   "Current medications are listed in the FHIR patient record.",
    },
    {
        "question":       "What was the billed amount and plan paid for claim CLM-10001?",
        "member_id":      None,
        "ground_truth":   "The EOB shows the billed amount and the plan paid amount.",
    },
    {
        "question":       "What is the medical justification for total knee arthroplasty?",
        "member_id":      None,
        "ground_truth":   "Medical justification includes confirmed osteoarthritis diagnosis, failed conservative treatment, and functional impairment.",
    },
    {
        "question":       "Was the appeal for a denied claim overturned or upheld?",
        "member_id":      None,
        "ground_truth":   "The appeal decision is documented in the appeal letter as either overturned or upheld.",
    },
    {
        "question":       "What prior authorization was submitted for procedure 27130?",
        "member_id":      None,
        "ground_truth":   "A prior authorization was submitted for CPT-27130 total hip arthroplasty.",
    },
    {
        "question":       "What is the patient diagnosis for a knee arthroscopy claim?",
        "member_id":      None,
        "ground_truth":   "The diagnosis is a derangement of the meniscus (ICD-10 M23.200).",
    },
    {
        "question":       "What clinical criteria were met for the prior authorization?",
        "member_id":      None,
        "ground_truth":   "Criteria include confirmed diagnosis, failed conservative treatment exceeding 3 months, documented functional impairment, and medical fitness.",
    },
    {
        "question":       "What is the allowed amount compared to billed for CPT-93306?",
        "member_id":      None,
        "ground_truth":   "The allowed amount is typically 55-75% of the billed amount as shown in the EOB.",
    },
]


def run_single_query(question: str, member_id: str = None) -> dict:
    chunks = vector_search(
        question  = question,
        top_k     = 4,
        member_id = member_id,
    )

    if not chunks:
        return {
            "answer":   "No relevant documents found.",
            "contexts": [],
            "chunks":   [],
        }

    messages   = build_prompt(question=question, retrieved_chunks=chunks)
    llm_result = generate(messages)

    return {
        "answer":   llm_result["answer"],
        "contexts": [c["text"] for c in chunks],
        "chunks":   chunks,
    }


def build_ragas_dataset(test_questions: list) -> Dataset:
    questions     = []
    answers       = []
    contexts      = []
    ground_truths = []

    print(f"\nRunning {len(test_questions)} test questions through the pipeline...")

    for i, item in enumerate(test_questions, 1):
        print(f"  [{i}/{len(test_questions)}] {item['question'][:70]}...")

        result = run_single_query(
            question  = item["question"],
            member_id = item.get("member_id"),
        )

        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })


def run_evaluation(sample: int = None):
    test_set = TEST_QUESTIONS
    if sample:
        test_set = test_set[:sample]

    dataset = build_ragas_dataset(test_set)

    print("\nRunning RAGAS evaluation...")
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    scores = {
        "faithfulness":      float(results["faithfulness"]),
        "answer_relevancy":  float(results["answer_relevancy"]),
        "context_precision": float(results["context_precision"]),
        "context_recall":    float(results["context_recall"]),
    }

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("uhg-rag-evaluation")

    run_name = f"ragas_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("num_questions",    len(test_set))
        mlflow.log_param("embedding_model",  "all-MiniLM-L6-v2")
        mlflow.log_param("chunk_size",       512)
        mlflow.log_param("chunk_overlap",    50)
        mlflow.log_param("top_k",            4)
        for metric, score in scores.items():
            mlflow.log_metric(metric, score)

    output_path = EVAL_DIR / f"ragas_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump({
            "run_name":  run_name,
            "timestamp": datetime.now().isoformat(),
            "scores":    scores,
            "num_questions": len(test_set),
        }, f, indent=2)

    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    for metric, score in scores.items():
        bar = "#" * int(score * 20)
        print(f"  {metric:<22} {score:.4f}  [{bar:<20}]")
    print(f"\n  Results saved to: {output_path}")
    print(f"  MLflow run: {run_name}")
    print("=" * 50)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS evaluation for UHG RAG pipeline")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of questions to evaluate (default: all)")
    args = parser.parse_args()
    run_evaluation(sample=args.sample)
