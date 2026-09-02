"""
main.py — Simple, no-Docker RAG demo with a Retrieval Grader validation step

What it does:
1) Embeds 5 sample documents in memory using Gemini (no database, no Docker).
2) For each query: retrieves the most similar doc (cosine similarity),
   grades its relevance, and only generates an answer if it's relevant.
3) If irrelevant -> flags "Transformation Required" and blocks generation.

Setup (run once):
  pip install google-genai python-dotenv
  Create a .env file in this folder with:
      GOOGLE_API_KEY=your-key-here
  (Get a free key at https://aistudio.google.com/apikey)

Run:
  python main.py
"""

import os
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
GEMINI_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "gemini-embedding-001"


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    embedding: List[float] = field(default_factory=list)


SAMPLE_DOCS: List[Document] = [
    Document("d1", "Corrective RAG Overview",
        "Corrective RAG (CRAG) adds a retrieval evaluator that grades retrieved documents. "
        "If context is incorrect, insufficient, or ambiguous, the system triggers corrective actions such as query rewrite "
        "or external search before generation. This prevents the model from generating from bad context."),
    Document("d2", "Self-RAG Overview",
        "Self-RAG validates the generated answer against retrieved sources. The model critiques its own output and checks groundedness. "
        "If unsupported by sources, it can regenerate, refuse, or tighten prompts. Self-RAG is post-generation validation."),
    Document("d3", "Agentic RAG Pattern",
        "Agentic RAG turns retrieval into a tool used by an agent. The agent decomposes complex queries into sub-queries, "
        "retrieves in multiple hops, and synthesizes results. This helps for multi-part questions and multi-step reasoning."),
    Document("d4", "RAG Evaluation: The RAG Triad",
        "RAG evaluation can measure faithfulness (answer supported by sources), answer relevance (answers the question), "
        "and context precision (retrieved documents are relevant). A/B testing compares variants on the same query set."),
    Document("d5", "Embedding Strategy Tradeoffs",
        "Embedding service choice involves cost, performance, and control. External APIs offer high quality but usage-based costs and limited control. "
        "Local models provide privacy and control but require infrastructure and maintenance. Evaluate end-to-end via A/B tests."),
]


# -----------------------------
# Retrieval (in-memory, no database)
# -----------------------------

def embed_text(text: str, task_type: str) -> List[float]:
    resp = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return resp.embeddings[0].values


def build_index(docs: List[Document]) -> None:
    print(f"Embedding {len(docs)} sample docs...")
    for d in docs:
        d.embedding = embed_text(f"{d.title}\n{d.text}", "RETRIEVAL_DOCUMENT")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def retrieve(docs: List[Document], query: str, top_k: int = 1) -> List[Dict[str, Any]]:
    query_vec = embed_text(query, "RETRIEVAL_QUERY")
    scored = sorted(
        ((cosine_similarity(query_vec, d.embedding), d) for d in docs),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [
        {"doc_id": d.doc_id, "title": d.title, "text": d.text, "similarity": round(sim, 4)}
        for sim, d in scored[:top_k]
    ]


# -----------------------------
# Step 1: Retrieval Grader
# -----------------------------

def _safe_json_extract(text: str) -> dict:
    text = (text or "").strip().strip("`")
    if text.lower().startswith("json"):
        text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def grade_retrieval(question: str, documents: List[Dict[str, Any]]) -> bool:
    """
    Grades whether the top retrieved document is relevant to the question.
    Returns True for 'yes', False for 'no'.
    """
    if not documents:
        return False

    prompt = f"""
You are a grader assessing the relevance of a retrieved document to a user question.
User question: {question}
Retrieved document: {documents[0]['text']}
Is the document relevant to the question? Provide your answer as a JSON object with a single key 'relevance' and a value of 'yes' or 'no'.
"""

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    data = _safe_json_extract(resp.text)
    return str(data.get("relevance", "")).strip().lower() == "yes"


def generate_answer(question: str, documents: List[Dict[str, Any]]) -> str:
    context = "\n\n".join(f"{d['title']}\n{d['text']}" for d in documents)
    prompt = f"""
Answer the question using ONLY the context below. If it doesn't contain the
answer, say you don't have enough information.

Context:
{context}

Question: {question}
Answer:
"""
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return resp.text.strip()


# -----------------------------
# Step 2: Integrate the grader into the pipeline
# -----------------------------

def run_pipeline(docs: List[Document], question: str) -> None:
    print(f"\n{'=' * 70}\nQuestion: {question}\n{'-' * 70}")

    retrieved = retrieve(docs, question, top_k=1)
    top = retrieved[0]
    print(f"Top doc: [{top['doc_id']}] {top['title']}  (similarity: {top['similarity']})")

    is_relevant = grade_retrieval(question, retrieved)
    print("Retrieval grade:", "RELEVANT ✅" if is_relevant else "IRRELEVANT ❌")

    if not is_relevant:
        print("Flag: Transformation Required — blocking generation.")
        return

    answer = generate_answer(question, retrieved)
    print("\nAnswer:", answer)


def main() -> None:
    build_index(SAMPLE_DOCS)

    test_queries = [
        "What is the difference between Corrective RAG and Self-RAG?",
        "How do you evaluate a RAG pipeline?",
        "What is the best recipe for banana bread?",  # should be flagged irrelevant
    ]

    for q in test_queries:
        run_pipeline(SAMPLE_DOCS, q)

    print("\nType a question (or press Enter to quit):")
    while True:
        q = input("> ").strip()
        if not q:
            break
        run_pipeline(SAMPLE_DOCS, q)


if __name__ == "__main__":
    main()