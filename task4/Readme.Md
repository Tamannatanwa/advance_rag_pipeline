# A/B Test: Agentic RAG vs. Corrective RAG for Production

**Prepared for:** Engineering Leadership
**Assignment:** Coursera CON04116 — Advanced RAG Patterns, Module 2 Graded Assignment
**Author:** Tamanna

---

## 1. Introduction

This report compares two advanced Retrieval-Augmented Generation (RAG) patterns — **Agentic RAG** and **Corrective RAG** — to determine which is better suited for a production knowledge-base assistant.

- **Agentic RAG**: an LLM-driven agent plans which tool calls are needed to answer a question, executes a knowledge-base search tool, and synthesizes a final answer from the tool outputs. This pattern is flexible and can decompose multi-part questions into separate retrieval calls.
- **Corrective RAG**: a more constrained pipeline that retrieves context, has an LLM grade the relevance of that context (0–5), and — if the grade is below a threshold — rewrites the query and re-retrieves before generating the final answer. This pattern is designed to self-correct against weak initial retrieval.

Both patterns share the same underlying components to keep the comparison fair:
- **Embedding model:** `gemini-embedding-001`
- **Generation/judge model:** Gemini Flash (with automatic fallback across model name candidates in case of retirement)
- **Vector store:** an in-memory Python list with cosine similarity (no external vector DB), holding 8 sample documents about two fictional products (Product A and Product B) covering overviews, pricing, integrations, customization, and customer feedback.

---

## 2. Implementation & Testing Methodology

### 2.1 Retrieval mechanism
Both patterns call the same `retrieve()` function: the query is embedded, compared against all stored document embeddings via cosine similarity, and the top-K (K=4) most similar chunks are returned. This ensures any difference in results comes from the *pattern*, not the retrieval math.

### 2.2 Generation logic
- **Agentic RAG** uses a `SimpleReasoningAgent` that (1) asks the LLM to plan tool calls in JSON, (2) executes a `KnowledgeBaseSearch` tool for each planned sub-query, and (3) asks the LLM to synthesize a final answer strictly from the tool outputs.
- **Corrective RAG** retrieves once, grades relevance, and generates directly from context if the grade clears the threshold (3/5). No agent planning step is involved — it's a linear pipeline.

### 2.3 Validation / correction steps
- **Corrective RAG's** defining feature: if the relevance grade is below 3/5, the query is rewritten by the LLM into a more specific, keyword-rich version and re-retrieved (up to 2 attempts total) before generation.
- **Agentic RAG** has no explicit correction step — correction is implicit in the agent's ability to issue multiple, more targeted tool calls during planning.

### 2.4 Evaluation: RAG-Triad
Each answer, for both patterns, is scored by an LLM judge on three metrics (0–5 scale):
- **Faithfulness** — is the answer fully supported by the retrieved context, with no fabricated facts?
- **Answer Relevance** — does the answer directly address the user's question?
- **Context Precision** — is the retrieved context itself relevant to the question (versus padded with irrelevant chunks)?

### 2.5 Test query set
Six queries were designed to exercise different retrieval difficulty levels: a multi-part comparison query, straightforward single-fact lookups, and an implicit-reasoning query ("why might a power user outgrow Product A?").

### 2.6 Operational notes
The free-tier Gemini API quota constrained the run (rate limits of roughly 5–10 requests/minute depending on account tier). The script was hardened with exponential backoff, retry logic for both API-level (429) and network-level (dropped connection) errors, and incremental checkpointing so a mid-run failure doesn't lose completed results.

---

## 3. Quantitative Comparison

*(Based on results available at time of writing — 1 of 6 queries completed; full table to be populated once the remaining queries finish running.)*

| Query | Pattern | Faithfulness | Answer Relevance | Context Precision | Correction Attempts |
|---|---|---|---|---|---|
| Compare Product A and Product B, then summarize customer feedback for Product B. | Agentic RAG | 5/5 | 5/5 | 3/5 | — |
| Compare Product A and Product B, then summarize customer feedback for Product B. | Corrective RAG | 5/5 | 5/5 | 5/5 | 1 |

**Observation from this query:** both patterns produced fully faithful, fully relevant answers. The difference shows up in **context precision** — the judge marked Agentic RAG's retrieved context as containing "heavy duplication of several chunks," while Corrective RAG's single-pass retrieval (which happened to score above the relevance threshold, needing no rewrite) returned a cleaner, non-redundant context set.

**Pending:** averages across all 6 queries, plus the average number of correction attempts triggered by Corrective RAG, will be added once the full run completes. Re-run `ab_test_rag.py` (it will resume automatically from `ab_test_results_partial.json`) and paste the finished `ab_test_results.json` / `score.txt` back to fill this section in.

---

## 4. Cost-Benefit Analysis: Embedding Strategy

| Factor | `gemini-embedding-001` (API) | Local embedding model (e.g., open-source sentence-transformer) |
|---|---|---|
| **API cost** | Pay per token via Gemini API; free tier available but rate-limited (this run hit those limits repeatedly) | No per-call cost after initial setup; only compute/hosting cost |
| **Latency** | Network round-trip per embedding call; adds up when re-embedding on every corrective-query rewrite | Local inference, typically lower latency once loaded, no network dependency |
| **Retrieval accuracy** | Generally strong, well-tuned for semantic search out of the box | Varies by model; may need fine-tuning on domain data to match API-model quality |
| **Data privacy** | Document/query text leaves your infrastructure to Google's servers | Data stays on-premises/in your own cloud — better for sensitive or regulated content |
| **Operational overhead** | Minimal — no model hosting, but subject to third-party rate limits and quota changes | Requires hosting, GPU/CPU provisioning, and model maintenance |

**Takeaway:** the API approach is faster to stand up and requires no infrastructure, but this test run itself demonstrates its main production risk — free-tier rate limits directly throttled throughput and reliability. For a low-volume internal tool, the API is likely fine. For a customer-facing or high-QPS production system, a self-hosted embedding model (or a paid Gemini tier with a guaranteed quota) removes the rate-limit dependency and keeps sensitive document content in-house.

---

## 5. Recommendation

Based on the single completed comparison and the architecture of both patterns:

- **Corrective RAG** is the safer default for production: it achieved equal faithfulness/relevance with better context precision, and its explicit grade-and-retry step gives an auditable signal (the relevance score) for monitoring retrieval quality over time — useful for alerting when the knowledge base needs updating.
- **Agentic RAG** is worth keeping for more complex, multi-hop questions where an agent's ability to issue several distinct, targeted tool calls could outperform a single retrieve-and-correct pass — this needs more queries in the test set to confirm.
- **Final recommendation:** ship Corrective RAG as the primary pattern, and re-evaluate this recommendation once the full 6-query run (or a larger query set) is complete — one data point isn't enough to generalize confidently.
- **On embeddings:** stay on the Gemini API for now, but budget for either a paid-tier quota increase or a self-hosted fallback embedding model before scaling beyond prototype traffic, given the rate-limiting behavior observed firsthand during this test.

---

## 6. Submission Contents

- `ab_test_rag.py` — Python implementation of both RAG patterns, the A/B test runner, and the RAG-Triad evaluator.
- `ab_test_results.json` — full per-query results and judge scores (to be finalized after the complete run).
- `score.txt` — plain-text summary of average scores per pattern.
- `README.md` — this report.