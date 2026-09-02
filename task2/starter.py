"""
starter.py — Weaviate Starter Script (Schema + Sample Docs + Retrieval Test)

This is designed to be a Coursera-provided starter file.

What it does:
1) Connect to Weaviate (default: http://localhost:8080)
2) Create a NEW schema/class if it doesn't exist
3) Insert sample docs if class is empty
4) Run a few retrieval queries (nearText) to prove it works

Works with your docker-compose (text2vec-transformers + transformers inference).

Run:
  python starter.py

Env overrides (optional):
  WEAVIATE_URL=http://localhost:8080
  WEAVIATE_CLASS=StarterDoc
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080").rstrip("/")
WEAVIATE_CLASS = os.getenv("WEAVIATE_CLASS", "StarterDoc").strip()


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str = "sample"


SAMPLE_DOCS: List[Document] = [
    Document(
        doc_id="sd1",
        title="Corrective RAG (CRAG) — Quick Note",
        text=(
            "Corrective RAG adds a retrieval evaluator that grades retrieved documents. "
            "If context is incorrect, insufficient, or ambiguous, the system triggers a corrective action "
            "such as query rewrite or external search before generation."
        ),
    ),
    Document(
        doc_id="sd2",
        title="Self-RAG — Quick Note",
        text=(
            "Self-RAG validates the final generated answer against retrieved sources. "
            "If the answer is not grounded in the context, the system can regenerate or refuse."
        ),
    ),
    Document(
        doc_id="sd3",
        title="Agentic RAG — Quick Note",
        text=(
            "Agentic RAG turns retrieval into a tool used by an agent. "
            "The agent decomposes complex questions into sub-queries and retrieves in multiple hops."
        ),
    ),
    Document(
        doc_id="sd4",
        title="RAG Evaluation — The RAG Triad",
        text=(
            "The RAG Triad is a common evaluation frame: Context Precision, Faithfulness, Answer Relevance. "
            "A/B testing compares different RAG variants using the same query set."
        ),
    ),
]


# -----------------------------
# HTTP helpers (no external deps)
# -----------------------------

def http_json(method: str, path: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Any:
    url = f"{WEAVIATE_URL}{path}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(
        url=url,
        data=data,
        method=method.upper(),
        headers={"Content-Type": "application/json"},
    )
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
            version = meta.get("version")
            print(f"Weaviate is up. Version: {version}")
            return
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"Weaviate not reachable after {max_wait_s}s. Last error: {last_err}")


# -----------------------------
# Schema
# -----------------------------

def class_exists(class_name: str) -> bool:
    schema = http_json("GET", "/v1/schema")
    classes = schema.get("classes", []) if isinstance(schema, dict) else []
    return any(c.get("class") == class_name for c in classes)


def create_class_if_missing(class_name: str) -> None:
    """
    Creates a new class configured for text2vec-transformers.
    """
    if class_exists(class_name):
        print(f"Schema ok: class '{class_name}' already exists.")
        return

    payload = {
        "class": class_name,
        "description": "Starter documents for Weaviate retrieval exercises",
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {
            "text2vec-transformers": {
                "vectorizeClassName": False
            }
        },
        "properties": [
            {"name": "doc_id", "dataType": ["text"], "description": "Stable doc id"},
            {"name": "title", "dataType": ["text"], "description": "Document title"},
            {"name": "text", "dataType": ["text"], "description": "Document body"},
            {"name": "source", "dataType": ["text"], "description": "Source label"},
        ],
    }

    http_json("POST", "/v1/schema", payload)
    print(f"Created schema class '{class_name}'.")


# -----------------------------
# Data load
# -----------------------------

def count_objects(class_name: str) -> int:
    """
    Count objects using GraphQL Aggregate.
    """
    gql = {
        "query": f"""
        {{
          Aggregate {{
            {class_name} {{
              meta {{ count }}
            }}
          }}
        }}
        """
    }
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

    objects = []
    for d in docs:
        objects.append({
            "class": class_name,
            "properties": {
                "doc_id": d.doc_id,
                "title": d.title,
                "text": d.text,
                "source": d.source,
            },
        })

    resp = http_json("POST", "/v1/batch/objects", {"objects": objects})

    # Detect batch errors
    errors = []
    if isinstance(resp, dict) and resp.get("errors"):
        errors.append(resp["errors"])
    if isinstance(resp, dict) and "result" in resp:
        for r in resp["result"]:
            if r.get("result", {}).get("errors"):
                errors.append(r["result"]["errors"])

    if errors:
        raise RuntimeError(f"Batch insert had errors: {json.dumps(errors)[:2000]}")

    # Give vectorizer a moment
    time.sleep(0.5)
    print("Insert complete.")


# -----------------------------
# Retrieval test
# -----------------------------

def near_text_search(class_name: str, query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    nearText vector search via GraphQL.
    """
    safe_query = query.replace('"', '\\"')  # avoid breaking GraphQL string
    gql = {
        "query": f"""
        {{
          Get {{
            {class_name}(
              nearText: {{ concepts: ["{safe_query}"] }}
              limit: {limit}
            ) {{
              doc_id
              title
              text
              source
              _additional {{
                distance
                id
              }}
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


def print_results(query: str, results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("-" * 80)
    if not results:
        print("No results.")
        print("=" * 80)
        return

    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        doc_id = r.get("doc_id", "")
        add = r.get("_additional") or {}
        dist = add.get("distance") if isinstance(add, dict) else None
        snippet = (r.get("text") or "")[:220].replace("\n", " ").strip()

        print(f"{i}. [{doc_id}] {title}")
        if dist is not None:
            print(f"   distance: {dist}")
        print(f"   {snippet}...")

    print("=" * 80)


def main() -> None:
    print(f"WEAVIATE_URL={WEAVIATE_URL}")
    print(f"WEAVIATE_CLASS={WEAVIATE_CLASS}")

    wait_for_weaviate(max_wait_s=60)
    create_class_if_missing(WEAVIATE_CLASS)
    insert_docs_if_empty(WEAVIATE_CLASS, SAMPLE_DOCS)

    demo_queries = [
        "What is Corrective RAG?",
        "How does Self-RAG work?",
        "Why do we use Agentic RAG?",
        "How do we evaluate RAG systems?",
    ]

    for q in demo_queries:
        res = near_text_search(WEAVIATE_CLASS, q, limit=3)
        print_results(q, res)

    print("\nType a query to test retrieval (or press Enter to exit):")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            break
        res = near_text_search(WEAVIATE_CLASS, q, limit=5)
        print_results(q, res)


if __name__ == "__main__":
    main()