"""
main.py — Weaviate + text2vec-transformers (docker-compose) self-contained demo

What it does:
1) Connects to Weaviate at http://localhost:8080
2) Creates schema (collection/class) if it doesn't exist
3) Inserts sample documents if class is empty
4) Runs retrieval for a few sample queries (and optional interactive mode)

Requirements:
- Your docker-compose services running:
    docker compose up -d
- Python 3.10+
- No extra pip deps needed (uses urllib only)

Run:
  python main.py

Env overrides (optional):
  WEAVIATE_URL=http://localhost:8080
  WEAVIATE_CLASS=CourseDoc
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080").rstrip("/")
WEAVIATE_CLASS = os.getenv("WEAVIATE_CLASS", "CourseDoc").strip()


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str = "sample"


SAMPLE_DOCS: List[Document] = [
    Document(
        doc_id="d1",
        title="Corrective RAG Overview",
        text=(
            "Corrective RAG (CRAG) adds a retrieval evaluator that grades retrieved documents. "
            "If context is incorrect, insufficient, or ambiguous, the system triggers corrective actions such as query rewrite "
            "or external search before generation. This prevents the model from generating from bad context."
        ),
    ),
    Document(
        doc_id="d2",
        title="Self-RAG Overview",
        text=(
            "Self-RAG validates the generated answer against retrieved sources. The model critiques its own output and checks groundedness. "
            "If unsupported by sources, it can regenerate, refuse, or tighten prompts. Self-RAG is post-generation validation."
        ),
    ),
    Document(
        doc_id="d3",
        title="Agentic RAG Pattern",
        text=(
            "Agentic RAG turns retrieval into a tool used by an agent. The agent decomposes complex queries into sub-queries, "
            "retrieves in multiple hops, and synthesizes results. This helps for multi-part questions and multi-step reasoning."
        ),
    ),
    Document(
        doc_id="d4",
        title="RAG Evaluation: The RAG Triad",
        text=(
            "RAG evaluation can measure faithfulness (answer supported by sources), answer relevance (answers the question), "
            "and context precision (retrieved documents are relevant). A/B testing compares variants on the same query set."
        ),
    ),
    Document(
        doc_id="d5",
        title="Embedding Strategy Tradeoffs",
        text=(
            "Embedding service choice involves cost, performance, and control. External APIs offer high quality but usage-based costs and limited control. "
            "Local models provide privacy and control but require infrastructure and maintenance. Evaluate end-to-end via A/B tests."
        ),
    ),
]


# -----------------------------
# HTTP helpers (no dependencies)
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
# Schema management (v1/schema)
# -----------------------------

def class_exists(class_name: str) -> bool:
    schema = http_json("GET", "/v1/schema")
    classes = schema.get("classes", []) if isinstance(schema, dict) else []
    return any(c.get("class") == class_name for c in classes)


def create_class_if_missing(class_name: str) -> None:
    """
    Creates a Weaviate class using text2vec-transformers module (as per your docker-compose).

    Notes:
    - vectorizer is set to "text2vec-transformers" (matches DEFAULT_VECTORIZER_MODULE)
    - no explicit moduleConfig needed if defaults are set, but we add it for clarity
    """
    if class_exists(class_name):
        print(f"Schema ok: class '{class_name}' already exists.")
        return

    payload = {
        "class": class_name,
        "description": "Course documents for RAG retrieval exercises",
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {
            "text2vec-transformers": {
                # Weaviate will call the inference container via TRANSFORMERS_INFERENCE_API
                "vectorizeClassName": False
            }
        },
        "properties": [
            {"name": "doc_id", "dataType": ["text"], "description": "Stable doc id"},
            {"name": "title", "dataType": ["text"], "description": "Title"},
            {"name": "text", "dataType": ["text"], "description": "Body text"},
            {"name": "source", "dataType": ["text"], "description": "Source label"},
        ],
    }

    http_json("POST", "/v1/schema", payload)
    print(f"Created schema class '{class_name}'.")


# -----------------------------
# Data loading
# -----------------------------

def count_objects(class_name: str) -> int:
    """
    Uses GraphQL aggregate to count objects. Works on 1.24.x.
    """
    query = {
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
    out = http_json("POST", "/v1/graphql", query)
    try:
        return int(out["data"]["Aggregate"][class_name][0]["meta"]["count"])
    except Exception:
        # If aggregate is unavailable for some reason, just return 0 to be safe.
        return 0


def insert_docs_if_empty(class_name: str, docs: List[Document]) -> None:
    existing = count_objects(class_name)
    if existing > 0:
        print(f"Data ok: class '{class_name}' already has {existing} objects. Skipping insert.")
        return

    print(f"Inserting {len(docs)} sample docs into '{class_name}'...")

    # Use batch endpoint
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

    payload = {"objects": objects}
    resp = http_json("POST", "/v1/batch/objects", payload)

    # Check for errors
    errors = []
    if isinstance(resp, dict) and "errors" in resp and resp["errors"]:
        errors.append(resp["errors"])
    if isinstance(resp, dict) and "result" in resp:
        for r in resp["result"]:
            if r.get("result", {}).get("errors"):
                errors.append(r["result"]["errors"])

    if errors:
        raise RuntimeError(f"Batch insert had errors: {json.dumps(errors)[:2000]}")

    # Give vectorizer a moment to finish
    time.sleep(0.5)
    print("Insert complete.")


# -----------------------------
# Retrieval
# -----------------------------

def near_text_search(class_name: str, query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Uses GraphQL Get + nearText (vector search).
    """
    # Escape quotes BEFORE f-string (important!)
    safe_query = query.replace('"', '\\"')

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
        return

    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        doc_id = r.get("doc_id", "")
        dist = None
        add = r.get("_additional") or {}
        if isinstance(add, dict):
            dist = add.get("distance")
        snippet = (r.get("text") or "")[:220].replace("\n", " ").strip()
        print(f"{i}. [{doc_id}] {title}")
        if dist is not None:
            print(f"   distance: {dist}")
        print(f"   {snippet}...")
    print("=" * 80)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    print(f"WEAVIATE_URL={WEAVIATE_URL}")
    print(f"WEAVIATE_CLASS={WEAVIATE_CLASS}")

    wait_for_weaviate(max_wait_s=60)
    create_class_if_missing(WEAVIATE_CLASS)
    insert_docs_if_empty(WEAVIATE_CLASS, SAMPLE_DOCS)

    demo_queries = [
        "What is the difference between Corrective RAG and Self-RAG?",
        "How do you evaluate a RAG pipeline?",
        "What is Agentic RAG and why is it useful?",
        "How do you choose an embedding service?",
    ]

    for q in demo_queries:
        res = near_text_search(WEAVIATE_CLASS, q, limit=3)
        print_results(q, res)

    # Optional interactive mode
    print("\nType a query for retrieval (or press Enter to exit):")
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
