"""
agentic_rag.py — starter.py, modified to implement the Agentic RAG pattern.
LLM used for generation + evaluation: Google Gemini (free tier).

Setup:
  pip install google-genai python-dotenv --break-system-packages
  export GEMINI_API_KEY="your-key-here"
  docker-compose up -d
  python agentic_rag.py
"""

from __future__ import annotations

import os
import re
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from google import genai
from dotenv import load_dotenv

load_dotenv()  # reads GEMINI_API_KEY (and others) from a .env file in the current folder

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080").rstrip("/")
WEAVIATE_CLASS = os.getenv("WEAVIATE_CLASS", "StarterDoc").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
DISTANCE_OK_THRESHOLD = 0.75
MAX_HOPS = 2

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY environment variable before running.")
_gemini_client = genai.Client(api_key=GEMINI_API_KEY)


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str = "sample"


SAMPLE_DOCS: List[Document] = [
    Document("sd1", "Corrective RAG (CRAG) — Quick Note",
             "Corrective RAG adds a retrieval evaluator that grades retrieved documents. "
             "If context is incorrect, insufficient, or ambiguous, the system triggers a corrective action "
             "such as query rewrite or external search before generation."),
    Document("sd2", "Self-RAG — Quick Note",
             "Self-RAG validates the final generated answer against retrieved sources. "
             "If the answer is not grounded in the context, the system can regenerate or refuse."),
    Document("sd3", "Agentic RAG — Quick Note",
             "Agentic RAG turns retrieval into a tool used by an agent. "
             "The agent decomposes complex questions into sub-queries and retrieves in multiple hops."),
    Document("sd4", "RAG Evaluation — The RAG Triad",
             "The RAG Triad is a common evaluation frame: Context Precision, Faithfulness, Answer Relevance. "
             "A/B testing compares different RAG variants using the same query set."),
]


# -----------------------------
# HTTP helpers (unchanged from provided starter.py)
# -----------------------------

def http_json(method: str, path: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Any:
    url = f"{WEAVIATE_URL}{path}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(url=url, data=data, method=method.upper(), headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            if not body:
                return None
            return json.loads(body)
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} {method} {path}: {body or e.reason}")
    except URLError as e:
        raise RuntimeError(f"Connection error calling {url}: {e}")


def wait_for_weaviate(max_wait_s: int = 60) -> None:
    start = time.time()
    last_err = None
    while time.time() - start < max_wait_s:
        try:
            meta = http_json("GET", "/v1/meta")
            print(f"Weaviate is up. Version: {meta.get('version')}")
            return
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"Weaviate not reachable after {max_wait_s}s. Last error: {last_err}")


def class_exists(class_name: str) -> bool:
    schema = http_json("GET", "/v1/schema")
    classes = schema.get("classes", []) if isinstance(schema, dict) else []
    return any(c.get("class") == class_name for c in classes)


def create_class_if_missing(class_name: str) -> None:
    if class_exists(class_name):
        print(f"Schema ok: class '{class_name}' already exists.")
        return
    payload = {
        "class": class_name,
        "description": "Starter documents for Weaviate retrieval exercises",
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {"text2vec-transformers": {"vectorizeClassName": False}},
        "properties": [
            {"name": "doc_id", "dataType": ["text"], "description": "Stable doc id"},
            {"name": "title", "dataType": ["text"], "description": "Document title"},
            {"name": "text", "dataType": ["text"], "description": "Document body"},
            {"name": "source", "dataType": ["text"], "description": "Source label"},
        ],
    }
    http_json("POST", "/v1/schema", payload)
    print(f"Created schema class '{class_name}'.")


def count_objects(class_name: str) -> int:
    gql = {"query": f"{{ Aggregate {{ {class_name} {{ meta {{ count }} }} }} }}"}
    out = http_json("POST", "/v1/graphql", gql)
    try:
        return int(out["data"]["Aggregate"][class_name][0]["meta"]["count"])
    except Exception:
        return 0


def insert_docs_if_empty(class_name: str, docs: List[Document]) -> None:
    existing = count_objects(class_name)
    if existing > 0:
        print(f"Data ok: class '{class_name}' already has {existing} objects. Skipping insert.")
        return
    print(f"Inserting {len(docs)} sample docs into '{class_name}'...")
    objects = [
        {"class": class_name, "properties": {"doc_id": d.doc_id, "title": d.title, "text": d.text, "source": d.source}}
        for d in docs
    ]
    resp = http_json("POST", "/v1/batch/objects", {"objects": objects})
    errors = []
    if isinstance(resp, dict) and resp.get("errors"):
        errors.append(resp["errors"])
    if isinstance(resp, dict) and "result" in resp:
        for r in resp["result"]:
            if r.get("result", {}).get("errors"):
                errors.append(r["result"]["errors"])
    if errors:
        raise RuntimeError(f"Batch insert had errors: {json.dumps(errors)[:2000]}")
    time.sleep(0.5)
    print("Insert complete.")


def near_text_search(class_name: str, query: str, limit: int = 3) -> List[Dict[str, Any]]:
    safe_query = query.replace('"', '\\"')
    gql = {
        "query": f"""
        {{
          Get {{
            {class_name}(
              nearText: {{ concepts: ["{safe_query}"] }}
              limit: {limit}
            ) {{
              doc_id title text source
              _additional {{ distance id }}
            }}
          }}
        }}
        """
    }
    out = http_json("POST", "/v1/graphql", gql)
    try:
        return out["data"]["Get"][class_name]
    except Exception:
        return []


# -----------------------------
# Gemini helper
# -----------------------------

def call_gemini(prompt: str) -> str:
    response = _gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return (response.text or "").strip()


def parse_json_response(raw: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```json|^```|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


# -----------------------------
# 1) Decision-making function
# -----------------------------

def classify_query(query: str) -> str:
    q = query.strip().lower()
    if q.count("?") > 1 or len(q.split()) > 14:
        return "complex"
    if re.search(r"\bcompare\b|\bvs\b|\bversus\b|\band\b.*\band\b", q):
        return "complex"
    return "simple"


def breakdown_query(query: str) -> List[str]:
    if classify_query(query) == "simple":
        return [query]
    parts = re.split(r"\bcompare\b|\bversus\b|\bvs\b|\band\b|\?", query, flags=re.IGNORECASE)
    parts = [p.strip(" ,.") for p in parts if p.strip(" ,.")]
    return parts if len(parts) > 1 else [query]


# -----------------------------
# 2) Agentic retrieval (dynamic, reasoning-driven — not a fixed loop)
# -----------------------------

def retrieve_documents(sub_query: str, limit: int = 3) -> List[Dict[str, Any]]:
    query, hops, results = sub_query, 0, []
    while hops < MAX_HOPS:
        hops += 1
        results = near_text_search(WEAVIATE_CLASS, query, limit=limit)
        best_distance = min((r.get("_additional", {}).get("distance", 1.0) for r in results), default=1.0)
        if results and best_distance <= DISTANCE_OK_THRESHOLD:
            break  # good enough, stop retrieving
        query = re.sub(r"\b(what|is|the|how|does|do|why)\b", "", query, flags=re.IGNORECASE).strip()
        if not query:
            break
    return results


def generate_answer(retrieved_docs: List[Dict[str, Any]], question: str) -> str:
    if not retrieved_docs:
        return "I couldn't find relevant information in the knowledge base."

    seen, contexts = set(), []
    for d in retrieved_docs:
        if d.get("doc_id") in seen:
            continue
        seen.add(d.get("doc_id"))
        contexts.append(f"[{d.get('title', '')}] {d.get('text', '')}")
    context_block = "\n\n".join(contexts)

    prompt = f"""Answer the question using ONLY the context below. Be concise (2-4 sentences).
If the context doesn't fully cover the question, answer with what it does cover.

Context:
{context_block}

Question: {question}

Answer:"""
    return call_gemini(prompt)


def agentic_rag(query: str) -> Dict[str, Any]:
    sub_queries = breakdown_query(query)
    retrieved_docs = [doc for sub_query in sub_queries for doc in retrieve_documents(sub_query)]
    answer = generate_answer(retrieved_docs, query)
    return {"question": query, "contexts": [d.get("text", "") for d in retrieved_docs], "answer": answer}


# -----------------------------
# 3) Evaluation — RAG Triad, judged by Gemini
# -----------------------------

def judge_metric(prompt: str, metric_key: str, model: str = GEMINI_MODEL) -> Dict[str, Any]:
    raw = call_gemini(prompt)
    try:
        parsed = parse_json_response(raw)
        return {
            "verdict": parsed.get("verdict", "no"),
            "score_0_5": float(parsed.get("score_0_5", 0)),
            "rationale": parsed.get("rationale", ""),
        }
    except Exception:
        return {"verdict": "no", "score_0_5": 0.0, "rationale": f"Could not parse judge output: {raw[:200]}"}


def evaluate_rag_triad(question: str, contexts: List[str], answer: str, model: str = GEMINI_MODEL) -> Dict[str, Any]:
    context_block = "\n\n---\n\n".join(contexts) if contexts else "(no context)"

    faithfulness_prompt = f"""You are evaluating a RAG system.

Metric: Faithfulness
Definition: The answer must be fully supported by the provided context. If the answer adds facts not in context, it is not faithful.

Context:
{context_block}

Answer:
{answer}

Return ONLY JSON with these keys:
{{
  "verdict": "yes" or "no",
  "score_0_5": 0 to 5,
  "rationale": "one short sentence"
}}"""
    faithfulness = judge_metric(faithfulness_prompt, "faithfulness", model=model)

    relevance_prompt = f"""You are evaluating a RAG system.

Metric: Answer Relevance
Definition: The answer should directly address the user's question.

Question:
{question}

Answer:
{answer}

Return ONLY JSON:
{{
  "verdict": "yes" or "no",
  "score_0_5": 0 to 5,
  "rationale": "one short sentence"
}}"""
    answer_relevance = judge_metric(relevance_prompt, "answer_relevance", model=model)

    context_precision_prompt = f"""You are evaluating a RAG system.

Metric: Context Precision
Definition: The retrieved context should be relevant to the user's question. Irrelevant context lowers precision.

Question:
{question}

Context:
{context_block}

Scoring guide:
5 = context is highly relevant with minimal noise
3 = mix of relevant and some irrelevant
1 = mostly irrelevant
0 = totally irrelevant

Return ONLY JSON:
{{
  "verdict": "yes" or "no",
  "score_0_5": 0 to 5,
  "rationale": "one short sentence"
}}"""
    context_precision = judge_metric(context_precision_prompt, "context_precision", model=model)

    overall = round(
        (faithfulness["score_0_5"] + answer_relevance["score_0_5"] + context_precision["score_0_5"]) / 3.0,
        2,
    )
    return {
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_precision": context_precision,
        "overall_score_0_5": overall,
    }


def main() -> None:
    print(f"WEAVIATE_URL={WEAVIATE_URL}")
    print(f"WEAVIATE_CLASS={WEAVIATE_CLASS}")
    print(f"GEMINI_MODEL={GEMINI_MODEL}")

    wait_for_weaviate(max_wait_s=60)
    create_class_if_missing(WEAVIATE_CLASS)
    insert_docs_if_empty(WEAVIATE_CLASS, SAMPLE_DOCS)

    query = "Compare Corrective RAG and Self-RAG, and how do we evaluate them?"
    result = agentic_rag(query)
    print(f"\nQuery: {query}")
    print(f"Answer: {result['answer']}")

    scores = evaluate_rag_triad(result["question"], result["contexts"], result["answer"])
    print(f"\nFaithfulness: {scores['faithfulness']['score_0_5']} — {scores['faithfulness']['rationale']}")
    print(f"Answer Relevance: {scores['answer_relevance']['score_0_5']} — {scores['answer_relevance']['rationale']}")
    print(f"Context Precision: {scores['context_precision']['score_0_5']} — {scores['context_precision']['rationale']}")
    print(f"Overall Score (0-5): {scores['overall_score_0_5']}")

    with open("score.txt", "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n")
        f.write(f"Answer: {result['answer']}\n\n")
        f.write(f"Faithfulness: {scores['faithfulness']['score_0_5']}\n")
        f.write(f"Answer Relevance: {scores['answer_relevance']['score_0_5']}\n")
        f.write(f"Context Precision: {scores['context_precision']['score_0_5']}\n")
        f.write(f"Overall Score (0-5): {scores['overall_score_0_5']}\n")
    print("\nSaved score.txt")


if __name__ == "__main__":
    main()