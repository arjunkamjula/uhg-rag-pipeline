"""
retrieval/llm_client.py

LLM client supporting Groq (default) and OpenAI.

Both providers use the OpenAI-compatible SDK — swapping between them
is a single environment variable change (LLM_PROVIDER=openai).
temperature=0.1 gives deterministic, factual answers suited to healthcare.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

GROQ_MODELS = {
    "default": "llama-3.3-70b-versatile",
    "fast":    "llama-3.1-8b-instant",
}
OPENAI_MODELS = {
    "default": "gpt-4o",
    "fast":    "gpt-4o-mini",
}


def get_client() -> OpenAI:
    if LLM_PROVIDER == "groq":
        return OpenAI(
            api_key  = os.getenv("GROQ_API_KEY"),
            base_url = "https://api.groq.com/openai/v1",
        )
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate(
    messages:    list,
    temperature: float = 0.1,
    max_tokens:  int   = 1024,
    fast:        bool  = False,
) -> dict:
    client = get_client()
    model  = (
        GROQ_MODELS["fast"]    if LLM_PROVIDER == "groq"   and fast else
        GROQ_MODELS["default"] if LLM_PROVIDER == "groq"            else
        OPENAI_MODELS["fast"]  if fast                               else
        OPENAI_MODELS["default"]
    )

    response = client.chat.completions.create(
        model       = model,
        messages    = messages,
        temperature = temperature,
        max_tokens  = max_tokens,
    )

    return {
        "answer":            response.choices[0].message.content,
        "model":             model,
        "provider":          LLM_PROVIDER,
        "prompt_tokens":     response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens":      response.usage.total_tokens,
    }
