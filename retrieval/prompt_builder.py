"""
retrieval/prompt_builder.py

Prompt assembly for the LLM generation step.

The system prompt enforces HIPAA-safe behavior:
  - Answer only from provided context, never from training knowledge
  - Say "not found in records" rather than guessing
  - Always cite source document and chunk index
  - Flag low-confidence answers when similarity scores are below 0.5
  - Never speculate on clinical or coverage decisions

temperature=0.1 is set in llm_client.py for deterministic, factual answers.
"""

SYSTEM_PROMPT = """You are a clinical claims intelligence assistant for UnitedHealthcare.
Your role is to help claims processors, prior authorization nurses, and appeals analysts
find information from member documents quickly and accurately.

RULES — follow these exactly:

1. Answer ONLY using the context documents provided below. Never use outside knowledge.
2. If the answer is not in the provided context, respond with:
   "This information was not found in the available records for this query."
3. Always cite your source at the end of your answer:
   Source: [filename] (chunk [n])
4. If multiple documents support the answer, cite all of them.
5. Never speculate on whether a claim should be approved or denied.
6. Never provide medical advice or clinical recommendations.
7. For dollar amounts, state exactly what the document says.
8. If context documents contradict each other, note the discrepancy.
9. Keep answers concise and factual. Use bullet points for multi-part answers.
10. If similarity scores are below 0.5, prefix your answer with [LOW CONFIDENCE].

You are operating in a HIPAA-regulated environment.
Accuracy and auditability are critical — every answer must be traceable to a source."""


def build_prompt(
    question:         str,
    retrieved_chunks: list,
    route_info:       dict = None,
) -> list:
    if retrieved_chunks:
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"--- Document {i} ---\n"
                f"Type: {chunk['doc_type']}\n"
                f"File: {chunk['source_file']}\n"
                f"Member ID: {chunk.get('member_id', 'N/A')}\n"
                f"Claim ID: {chunk.get('claim_id', 'N/A')}\n"
                f"Date: {chunk.get('date', 'N/A')}\n"
                f"Relevance score: {chunk['score']}\n"
                f"Content:\n{chunk['text']}\n"
            )
        context_block = "\n".join(context_parts)
    else:
        context_block = "No relevant documents were retrieved for this query."

    route_note = ""
    if route_info:
        if route_info.get("confidence") == "low":
            route_note = (
                "\n[NOTE: This query had no specific member or claim identifiers. "
                "Results may span multiple members. Verify relevance carefully.]\n"
            )
        elif route_info.get("member_id"):
            route_note = (
                f"\n[NOTE: Results filtered to member {route_info['member_id']} only.]\n"
            )

    user_content = (
        f"{route_note}"
        f"CONTEXT DOCUMENTS:\n\n"
        f"{context_block}\n\n"
        f"QUESTION: {question}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def format_sources(chunks: list) -> list:
    return [
        {
            "source_file": chunk["source_file"],
            "doc_type":    chunk["doc_type"],
            "member_id":   chunk.get("member_id"),
            "claim_id":    chunk.get("claim_id"),
            "date":        chunk.get("date"),
            "score":       chunk["score"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in chunks
    ]
