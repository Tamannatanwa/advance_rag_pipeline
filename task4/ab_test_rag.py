"""
A/B Test: Agentic RAG vs Corrective RAG  (Docker-free, VS Code terminal only)
------------------------------------------------------------------------------
Coursera "Advanced RAG Patterns" - Module 2 Graded Assignment (CON04116)

No Weaviate, no Docker. Everything runs in one Python process:
  - Embeddings:  Gemini embedding API (gemini-embedding-001)
  - Vector store: a plain Python list + cosine similarity (in-memory)
  - Generation:   Gemini (same model used by BOTH patterns, for a fair test;
                   auto-falls back through MODEL_CANDIDATES if a model name
                   gets retired)
  - Judge:        Gemini, scoring the RAG-Triad (faithfulness, answer
                   relevance, context precision)

Setup:
    pip install -r requirements.txt

    # create a file named .env in the same folder with:
    GEMINI_API_KEY=your-key-here

Run:
    python3 ab_test_rag.py
"""

import os
import re
import json
import time
import statistics
from dataclasses import dataclass
from typing import Callable, List, Dict, Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors

# =========================================================
# CONFIG
# =========================================================
load_dotenv()  # reads GEMINI_API_KEY from a .env file in this folder
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

EMBED_MODEL = "gemini-embedding-001"
# Google renames/retires Flash models fairly often on the free tier, so we
# try these in order and stick with whichever one actually works.
MODEL_CANDIDATES = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-flash-latest"]
GEN_MODEL = MODEL_CANDIDATES[0]
JUDGE_MODEL = MODEL_CANDIDATES[0]
_working_model = None  # discovered on first successful call, then reused
TOP_K = 4
RELEVANCE_THRESHOLD = 3            # out of 5, used by Corrective RAG's grader

# Free-tier pacing: space out calls so we don't hit per-minute quotas.
MIN_SECONDS_BETWEEN_CALLS = 4.5
_last_call_time = 0.0


def _pace() -> None:
    """Sleep just enough to keep us under the free-tier requests-per-minute limit."""
    global _last_call_time
    now = time.time()
    wait = MIN_SECONDS_BETWEEN_CALLS - (now - _last_call_time)
    if wait > 0:
        time.sleep(wait)
    _last_call_time = time.time()


def _retry_delay_seconds(e: "errors.APIError", attempt: int) -> float:
    """Pull Google's suggested retry delay out of a 429 error, or fall back."""
    try:
        details = e.details.get("error", {}).get("details", [])
        for d in details:
            if d.get("@type", "").endswith("RetryInfo"):
                return float(str(d.get("retryDelay", "0")).rstrip("s")) + 2
    except Exception:
        pass
    return min(60, 10 * (attempt + 1))


# =========================================================
# GEMINI HELPERS
# =========================================================
def call_gemini(prompt: str, model: str = None, temperature: float = 0.0,
                 max_retries: int = 6) -> str:
    """Single shared LLM call so both patterns use the identical base model.
    Falls back through MODEL_CANDIDATES if a model has been retired (404)."""
    global _working_model
    candidates = []
    if _working_model:
        candidates.append(_working_model)
    if model:
        candidates.append(model)
    for c in MODEL_CANDIDATES:
        if c not in candidates:
            candidates.append(c)

    last_error = None
    for candidate in candidates:
        for attempt in range(max_retries):
            _pace()
            try:
                resp = client.models.generate_content(
                    model=candidate,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=temperature),
                )
                _working_model = candidate
                return (resp.text or "").strip()
            except errors.ClientError as e:
                if e.code == 429:
                    wait = _retry_delay_seconds(e, attempt)
                    print(f"  [rate limited] waiting {wait:.0f}s before retrying "
                          f"({attempt + 1}/{max_retries})...")
                    time.sleep(wait)
                    continue
                if e.code == 404:
                    print(f"  [model unavailable] {candidate} - trying next candidate...")
                    last_error = e
                    break  # try the next candidate model
                raise
        else:
            continue
    raise RuntimeError(
        "Could not reach any Gemini model in MODEL_CANDIDATES "
        f"({MODEL_CANDIDATES}). Last error: {last_error}"
    )


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT",
                max_retries: int = 6) -> List[float]:
    for attempt in range(max_retries):
        _pace()
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return resp.embeddings[0].values
        except errors.ClientError as e:
            if e.code == 429:
                wait = _retry_delay_seconds(e, attempt)
                print(f"  [rate limited] waiting {wait:.0f}s before retrying "
                      f"embedding ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Gemini embed_content kept getting rate-limited. "
                        "Try again in a minute.")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# =========================================================
# IN-MEMORY VECTOR STORE (replaces Weaviate/Docker)
# =========================================================
SAMPLE_DOCS = [
    ("Product A is a lightweight analytics dashboard aimed at small teams. "
     "It offers pre-built templates, a simple drag-and-drop editor, and "
     "one-click sharing, but has limited customization for power users.",
     "product_a_overview"),
    ("Product A pricing is a flat $29/month per workspace with unlimited "
     "seats. There are no enterprise tiers and no volume discounts.",
     "product_a_pricing"),
    ("Product B is an enterprise analytics platform with advanced "
     "customization, role-based access control, and an open API for "
     "integrations. Configuration requires more setup effort than Product A.",
     "product_b_overview"),
    ("Product B pricing scales with data volume and the number of advanced "
     "features enabled (custom dashboards, governance, and priority "
     "support), so exact pricing is not publicly listed and depends on the "
     "customer's usage tier.",
     "product_b_pricing"),
    ("Customer feedback for Product A: users frequently praise how quick "
     "it is to get started and how clean the default templates look, but "
     "some power users say they outgrow it once they need custom metrics.",
     "product_a_feedback"),
    ("Customer feedback for Product B: customers report that Product B can "
     "be complex to configure, especially when compared to Product A. "
     "Feedback frequently mentions its capabilities in dashboards, "
     "analytics, governance, and APIs. While it offers powerful features "
     "for enterprise needs, the complexity of setup and configuration is "
     "a common concern among users.",
     "product_b_feedback"),
    ("Both Product A and Product B support CSV and API-based data import. "
     "Product A only supports a handful of connectors; Product B supports "
     "over 50 third-party integrations out of the box.",
     "integration_comparison"),
    ("Product A has limited customization (power users want more options). "
     "Product B offers advanced features and deeper customization aimed at "
     "enterprise needs.",
     "customization_comparison"),
]

# Populated once at startup by build_vector_store()
VECTOR_STORE: List[Dict[str, Any]] = []


def build_vector_store() -> None:
    """Embeds every sample doc once and keeps them in memory for the run."""
    if VECTOR_STORE:
        return
    for text, source in SAMPLE_DOCS:
        VECTOR_STORE.append({
            "text": text,
            "source": source,
            "embedding": embed_text(text, task_type="RETRIEVAL_DOCUMENT"),
        })


def retrieve(query_text: str, k: int = TOP_K) -> List[str]:
    """Shared retrieval mechanism: cosine similarity over in-memory embeddings."""
    query_emb = embed_text(query_text, task_type="RETRIEVAL_QUERY")
    scored = [
        (cosine_similarity(query_emb, doc["embedding"]), doc["text"])
        for doc in VECTOR_STORE
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:k]]


# =========================================================
# PATTERN A: AGENTIC RAG
# (agent plans tool calls, executes them, then writes the final answer)
# =========================================================
@dataclass
class Tool:
    name: str
    func: Callable[[str], Any]
    description: str


class SimpleReasoningAgent:
    """
    A lightweight agent that:
      1) asks the LLM to decide which tool(s) to call and with what queries
      2) executes those tool calls
      3) asks the LLM to write the final answer using the tool outputs
    """

    def __init__(self, tools: List[Tool], model: str = GEN_MODEL):
        self.tools = {t.name: t for t in tools}
        self.model = model

    def plan(self, user_query: str) -> List[Dict[str, str]]:
        tool_descriptions = "\n".join(
            f'- "{t.name}": {t.description}' for t in self.tools.values()
        )
        prompt = f'''
You are planning tool calls to answer this user question:
{user_query}

Available tools:
{tool_descriptions}

Decide what tool calls are needed. Return ONLY JSON: a list of objects with keys:
- "tool": tool name
- "input": the query to send the tool

Rules:
- If the user asks to compare two things, do at least one tool call per thing.
- Keep tool inputs short and specific.
'''
        text = call_gemini(prompt, model=self.model)
        return self._parse_json_list(text)

    @staticmethod
    def _parse_json_list(text: str) -> List[Dict[str, str]]:
        cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return []

    def answer(self, user_query: str, tool_results: List[Dict[str, Any]]) -> str:
        context_block = "\n\n".join(
            f"[{r['tool']} | input: {r['input']}]\n{r['output'].get('answer', r['output'])}"
            for r in tool_results
        )
        prompt = f'''
Answer the user's question using ONLY the information in the tool results below.
If something isn't covered, say so plainly instead of guessing.

User question: {user_query}

Tool results:
{context_block}

Final answer:
'''
        return call_gemini(prompt, model=self.model)

    def run(self, user_query: str) -> Dict[str, Any]:
        plan = self.plan(user_query)
        if not plan:
            return {"plan": [], "tool_results": [], "contexts": [], "final": "I couldn't form a tool plan."}

        tool_results = []
        all_contexts = []
        for step in plan:
            tool_name = step.get("tool")
            tool_input = step.get("input", "")
            tool = self.tools.get(tool_name)
            if not tool:
                tool_results.append({"tool": tool_name, "input": tool_input,
                                      "output": {"answer": "Unknown tool"}})
                continue
            output = tool.func(tool_input)  # returns dict(question, contexts, answer)
            tool_results.append({"tool": tool_name, "input": tool_input, "output": output})
            all_contexts.extend(output.get("contexts", []))

        final = self.answer(user_query, tool_results)
        return {"plan": plan, "tool_results": tool_results, "contexts": all_contexts, "final": final}


def _knowledge_base_search(query: str) -> Dict[str, Any]:
    """The RAG 'tool' the agent calls: retrieve + generate over those contexts."""
    contexts = retrieve(query)
    context_block = "\n\n---\n\n".join(contexts) if contexts else "(no context)"
    prompt = f'''
Answer the question using only the context below.

Context:
{context_block}

Question: {query}

Answer:
'''
    answer = call_gemini(prompt)
    return {"question": query, "contexts": contexts, "answer": answer}


rag_tool = Tool(
    name="KnowledgeBaseSearch",
    func=_knowledge_base_search,
    description="Use this tool to answer questions about products, pricing, "
                "customization, integrations, and customer feedback.",
)


def run_agentic_rag(question: str) -> Dict[str, Any]:
    agent = SimpleReasoningAgent(tools=[rag_tool])
    result = agent.run(question)
    return {
        "question": question,
        "contexts": result["contexts"],
        "answer": result["final"],
    }


# =========================================================
# PATTERN B: CORRECTIVE RAG
# (retrieve -> grade relevance -> if weak, rewrite query & re-retrieve -> generate)
# =========================================================
def grade_relevance(question: str, contexts: List[str]) -> int:
    """LLM grader: how relevant is the retrieved context to the question? 0-5."""
    context_block = "\n\n---\n\n".join(contexts) if contexts else "(no context)"
    prompt = f'''
Rate how relevant the CONTEXT below is to answering the QUESTION, on a
scale of 0 (irrelevant) to 5 (fully relevant). Return ONLY the number.

Question: {question}

Context:
{context_block}
'''
    text = call_gemini(prompt)
    match = re.search(r"\d+", text)
    return int(match.group()) if match else 0


def rewrite_query(question: str) -> str:
    """Ask the LLM to rewrite the query for a stronger retrieval attempt."""
    prompt = f'''
The following search query returned weak or irrelevant results from a
document database. Rewrite it as a more specific, keyword-rich search
query that is more likely to retrieve relevant documents. Return ONLY
the rewritten query, nothing else.

Original query: {question}
'''
    return call_gemini(prompt).strip().strip('"')


def run_corrective_rag(question: str, max_attempts: int = 2) -> Dict[str, Any]:
    contexts = retrieve(question)
    score = grade_relevance(question, contexts)
    attempts = 1
    used_query = question

    while score < RELEVANCE_THRESHOLD and attempts < max_attempts:
        used_query = rewrite_query(used_query)
        contexts = retrieve(used_query)
        score = grade_relevance(question, contexts)
        attempts += 1

    context_block = "\n\n---\n\n".join(contexts) if contexts else "(no context)"
    prompt = f'''
Answer the question using only the context below. If the context is
insufficient, say so plainly instead of guessing.

Context:
{context_block}

Question: {question}

Answer:
'''
    answer = call_gemini(prompt)
    return {
        "question": question,
        "contexts": contexts,
        "answer": answer,
        "correction_attempts": attempts,
        "final_relevance_score": score,
    }


# =========================================================
# JUDGE: RAG-TRIAD (faithfulness, answer relevance, context precision)
# =========================================================
def judge_metric(prompt: str, model: str = JUDGE_MODEL) -> Dict[str, Any]:
    text = call_gemini(prompt, model=model)
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = {"verdict": "no", "score_0_5": 0, "rationale": f"Unparseable judge output: {text[:200]}"}
    return parsed


def evaluate_rag_triad(question: str, contexts: List[str], answer: str,
                        model: str = JUDGE_MODEL) -> Dict[str, Any]:
    context_block = "\n\n---\n\n".join(contexts) if contexts else "(no context)"

    faithfulness_prompt = f'''
You are evaluating a RAG system.

Metric: Faithfulness
Definition: The answer must be fully supported by the provided context. If
the answer adds facts not in context, that is a faithfulness failure.

Context:
{context_block}

Answer:
{answer}

Return ONLY JSON with these keys:
{{
  "verdict": "yes" or "no",
  "score_0_5": 0 to 5,
  "rationale": "one short sentence"
}}
'''
    faithfulness = judge_metric(faithfulness_prompt, model=model)

    relevance_prompt = f'''
You are evaluating a RAG system.

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
}}
'''
    answer_relevance = judge_metric(relevance_prompt, model=model)

    context_precision_prompt = f'''
You are evaluating a RAG system.

Metric: Context Precision
Definition: The retrieved context should be relevant to the user's question.
Irrelevant context lowers precision.

Question:
{question}

Context:
{context_block}

Scoring guide:
5 = fully relevant context
3 = partially relevant / misses an important part
1 = barely relevant
0 = not related

Return ONLY JSON:
{{
  "verdict": "yes" or "no",
  "score_0_5": 0 to 5,
  "rationale": "one short sentence"
}}
'''
    context_precision = judge_metric(context_precision_prompt, model=model)

    return {
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_precision": context_precision,
    }


# =========================================================
# A/B TEST RUNNER
# =========================================================
QUERY_SET = [
    "Compare Product A and Product B, then summarize customer feedback for Product B.",
    "What is the pricing model for Product A?",
    "Why might a power user outgrow Product A?",
    "How many third-party integrations does Product B support?",
    "What are the main customer complaints about Product B?",
    "Which product is easier to set up, A or B?",
]


def _avg(values: List[float]) -> float:
    return round(statistics.mean(values), 2) if values else 0.0


def run_ab_test() -> Dict[str, Any]:
    build_vector_store()

    rows = []
    for question in QUERY_SET:
        print(f"\n=== Query: {question} ===")

        a_result = run_agentic_rag(question)
        a_eval = evaluate_rag_triad(a_result["question"], a_result["contexts"], a_result["answer"])

        b_result = run_corrective_rag(question)
        b_eval = evaluate_rag_triad(b_result["question"], b_result["contexts"], b_result["answer"])

        rows.append({
            "question": question,
            "agentic_rag": {"answer": a_result["answer"], "eval": a_eval},
            "corrective_rag": {
                "answer": b_result["answer"],
                "eval": b_eval,
                "correction_attempts": b_result["correction_attempts"],
            },
        })

        print(f"  Agentic RAG   -> faithfulness={a_eval['faithfulness']['score_0_5']}, "
              f"relevance={a_eval['answer_relevance']['score_0_5']}, "
              f"context_precision={a_eval['context_precision']['score_0_5']}")
        print(f"  Corrective RAG-> faithfulness={b_eval['faithfulness']['score_0_5']}, "
              f"relevance={b_eval['answer_relevance']['score_0_5']}, "
              f"context_precision={b_eval['context_precision']['score_0_5']} "
              f"(correction attempts: {b_result['correction_attempts']})")

    summary = summarize(rows)
    save_results(rows, summary)
    return {"rows": rows, "summary": summary}


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def collect(pattern_key: str, metric_key: str) -> List[float]:
        return [r[pattern_key]["eval"][metric_key]["score_0_5"] for r in rows]

    summary = {}
    for pattern_key in ("agentic_rag", "corrective_rag"):
        summary[pattern_key] = {
            "avg_faithfulness": _avg(collect(pattern_key, "faithfulness")),
            "avg_answer_relevance": _avg(collect(pattern_key, "answer_relevance")),
            "avg_context_precision": _avg(collect(pattern_key, "context_precision")),
        }
    summary["corrective_rag"]["avg_correction_attempts"] = _avg(
        [r["corrective_rag"]["correction_attempts"] for r in rows]
    )
    return summary


def save_results(rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    with open("ab_test_results.json", "w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2)

    with open("score.txt", "w") as f:
        f.write("A/B TEST RESULTS: Agentic RAG vs Corrective RAG\n")
        f.write("=" * 55 + "\n\n")
        for pattern_key, label in (("agentic_rag", "Agentic RAG"), ("corrective_rag", "Corrective RAG")):
            s = summary[pattern_key]
            f.write(f"{label}\n")
            f.write(f"  Avg Faithfulness:       {s['avg_faithfulness']} / 5\n")
            f.write(f"  Avg Answer Relevance:   {s['avg_answer_relevance']} / 5\n")
            f.write(f"  Avg Context Precision:  {s['avg_context_precision']} / 5\n")
            if "avg_correction_attempts" in s:
                f.write(f"  Avg Correction Attempts:{s['avg_correction_attempts']}\n")
            f.write("\n")

    print("\nSaved detailed results to ab_test_results.json")
    print("Saved summary to score.txt")


if __name__ == "__main__":
    result = run_ab_test()
    print("\n=== SUMMARY ===")
    print(json.dumps(result["summary"], indent=2))