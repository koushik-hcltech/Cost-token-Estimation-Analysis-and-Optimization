# Cost/token Estimation, Analysis, and Optimization (Code Assistant Agent Lab Runbook)

This repository contains the complete runbook, reusable reference patterns, and solution artifacts for **measuring and optimizing the cost per query of a multi-agent application** built with the **Google Agent Development Kit (ADK)**, **Vertex AI Gemini 3 models**, **Agent Search (Discovery Engine)**, **ChromaDB**, and the **Vertex AI Gen AI Evaluation Service**.

---

## 1. Architecture Overview

```
                         ┌──────────────────────────────┐
                         │         User Query           │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │              Coordinator              │
                    │  (ADK Agent: gemini-3.5-flash / Pro)  │
                    │  - Decomposes developer query         │
                    │  - Invokes sub-agents via AgentTool   │
                    │  - Synthesizes cited final answer     │
                    └───────┬───────────────────────┬───────┘
                            │                       │
             AgentTool call │                       │ AgentTool call
                            ▼                       ▼
        ┌───────────────────────────────┐ ┌───────────────────────────────┐
        │           DocsAgent           │ │        CommunityAgent         │
        │  (ADK Sub-Agent: Flash-Lite)  │ │  (ADK Sub-Agent: Flash-Lite)  │
        │  - Calls docs_lookup(query)   │ │  - Calls community_lookup(q)  │
        └───────────────┬───────────────┘ └───────────────┬───────────────┘
                        │                                 │
                        ▼                                 ▼
        ┌───────────────────────────────┐ ┌───────────────────────────────┐
        │   Managed Agent Search (VAIS) │ │   Local ChromaDB Vector Store │
        │  - DataStore: code_assistant  │ │  - gemini-embedding-001       │
        │  - Engine: docs-engine        │ │  - Stack Overflow corpus      │
        │  - Billed per query ($1.50/1k)│ │  - Billed per embedding token │
        └───────────────────────────────┘ └───────────────────────────────┘
```

### Why Multi-Agent Costs Multiply Quickly
A single user question fans out into:
1. **Coordinator Turn 1**: Reads the user prompt + tool declarations, reasons (`thought_tokens`), and emits tool calls (`DocsAgent`, `CommunityAgent`).
2. **Sub-Agent Turn 1 (`DocsAgent` & `CommunityAgent`)**: Each sub-agent reads its instructions + query and emits a retrieval tool call (`docs_lookup`, `community_lookup`).
3. **Retrieval Calls**:
   - `DocsAgent` queries **Agent Search (Discovery Engine)** (billed per search request).
   - `CommunityAgent` embeds the query with `gemini-embedding-001` (billed per input token) and searches ChromaDB.
4. **Sub-Agent Turn 2**: Each sub-agent reads the retrieved `top_k` documents and summarizes the findings with `[doc_id]` citations.
5. **Coordinator Turn 2**: Reads the conversation history + both sub-agent responses, reasons (`thought_tokens`), and generates the final cited answer.

---

## 2. Step-by-Step Lab Execution Runbook

### Step 1: Environment & Dependency Setup
Install the required packages in an isolated virtual environment (or Workbench kernel):

```bash
python3 -m venv .venv
.venv/bin/pip install -q \
  "google-adk>=1.0.0" \
  "google-genai>=1.0.0" \
  "google-cloud-aiplatform[evaluation]>=1.70.0" \
  "google-cloud-discoveryengine>=0.13.0" \
  "google-cloud-storage" \
  "chromadb>=1.0.0" \
  numpy pandas matplotlib
```

Configure the Vertex AI environment variables (Gemini 3 models serve on the `global` endpoint, while the Evaluation Service runs in `us-central1`):

```python
import os

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
LOCATION = "global"           # Gemini 3 family serves from the global endpoint
EVAL_LOCATION = "us-central1" # Vertex Gen AI Evaluation Service is regional

os.environ.update(
    GOOGLE_GENAI_USE_VERTEXAI="TRUE",
    GOOGLE_CLOUD_PROJECT=PROJECT,
    GOOGLE_CLOUD_LOCATION=LOCATION,
)
```

---

### Step 2: Define the Cost Model & Usage Ledger
Every billable event (LLM input, cached input, output + reasoning/thought tokens, embedding tokens, and Agent Search queries) is priced via `cost_of(record)` and tracked via ADK callbacks in `UsageLedger`:

```python
PRICING = {
    # USD per 1M tokens:              input        cached       output (thoughts bill as output)
    "gemini-3.1-pro-preview": {"input": 2.00, "cached": 0.20, "output": 12.00},
    "gemini-3.5-flash":       {"input": 1.50, "cached": 0.15, "output":  9.00},
    "gemini-3.1-flash-lite":  {"input": 0.25, "cached": 0.025, "output": 1.50},
    "gemini-2.5-pro":         {"input": 1.25, "cached": 0.13, "output": 10.00},
    "gemini-2.5-flash":       {"input": 0.30, "cached": 0.03, "output":  2.50},
    "embedding":  {"per_1m_tokens": 0.15},    # gemini-embedding-001
    "vais_query": {"per_1k_queries": 1.50},   # Agent Search Standard Edition
}

def cost_of(record):
    """USD for one billable event. Thought (reasoning) tokens bill at the output rate."""
    ct = record.get("call_type")
    if ct == "model":
        p = PRICING[record["model"]]
        out = record.get("output_tokens", 0) + record.get("thought_tokens", 0)
        return round((record.get("input_tokens", 0) * p["input"]
                      + record.get("cached_tokens", 0) * p["cached"]
                      + out * p["output"]) / 1e6, 12)
    if ct == "embedding":
        return round(record.get("embed_tokens", 0) * PRICING["embedding"]["per_1m_tokens"] / 1e6, 12)
    if ct == "retrieval":
        return round(record.get("retrieval_queries", 0) * PRICING["vais_query"]["per_1k_queries"] / 1e3, 12)
    return 0.0
```

---

### Step 3: Provision Agent Search (Discovery Engine) & ChromaDB (Task 4)
Provision the managed **Agent Search** data store (`code_assistant-docs`) and search engine (`code_assistant-docs-engine`) in `global`, import the 20 documentation items from `assets/docs.jsonl`, and build the in-memory ChromaDB index for `assets/stackoverflow.jsonl`:

```python
from code_assistant import vais
from code_assistant.data import load_jsonl
from code_assistant.retrieval import build_index

VAIS_LOCATION = "global"
DS_ID = "code_assistant-docs"
ENGINE_ID = "code_assistant-docs-engine"

# Idempotently creates the DataStore, imports docs, creates the Search Engine, and waits until searchable
DOCS_SC = vais.ensure(PROJECT, VAIS_LOCATION, DS_ID, ENGINE_ID, load_jsonl("assets/docs.jsonl"))

# Build local ChromaDB collection for Stack Overflow community posts
so = load_jsonl("assets/stackoverflow.jsonl")
community = build_index(so, "gemini-embedding-001", PROJECT, LOCATION, name="community")
```

---

### Step 4: Apply the Three Cost-Optimization Levers

| Lever | Mechanism | Impact on Cost & Quality |
| :--- | :--- | :--- |
| **Lever 1a: Coordinator Model Tiering (Pro -> Flash)** | Switch Coordinator from `gemini-3.1-pro-preview` ($2/$12 per 1M) to `gemini-3.5-flash` ($1.50/$9 per 1M). | Reduces Coordinator token rates by ~25% while preserving high orchestration & synthesis quality (`>= 0.90`). |
| **Lever 1b: Coordinator Budget Tier (Flash-Lite)** | Switch Coordinator to `gemini-3.1-flash-lite` ($0.25/$1.50 per 1M). | **Anti-pattern for Coordinator**: Unreliable multi-agent orchestration, missed citations, and high variance. |
| **Sub-Agent Tiering (Flash -> Flash-Lite)** | Switch `DocsAgent` and `CommunityAgent` to `gemini-3.1-flash-lite` ($0.25/$1.50 per 1M). | **6x cheaper** sub-agent token rates! Because sub-agents only execute a single lookup tool and summarize snippets (with `thinking_level="MINIMAL"`), quality does not degrade. |
| **Lever 2: Context & Semantic Caching** | Context caching (`caches.create`) discounts repeated prefixes `>= 2,048` tokens by **90%**. Semantic caching embeds incoming queries and serves cached answers above a cosine similarity threshold (e.g., `0.85`). | Context caching applies when system prompts/manuals exceed 2,048 tokens. Semantic caching eliminates the entire multi-agent pipeline cost on repeated/paraphrased questions. |
| **Lever 3a: Reasoning Controls (`thinking_level`)** | Pin Coordinator `thinking_level="LOW"` instead of `"HIGH"`. | Dramatically cuts `thought_tokens` (which bill at the expensive output token rate) with minimal quality impact on grounded RAG synthesis. |
| **Lever 3b: Retrieval Breadth (`top_k`)** | Reduce retrieved chunks per tool call from `top_k=5` to `top_k=2`. | Trims input tokens passed to both the sub-agents and Coordinator. |

---

### Step 5: Final Challenge Configuration & Evaluation (Task 5)

#### Production Constraints
- **Cost Budget**: Average cost per query `<= $0.0135` USD on the 16 held-out queries (`assets/eval_holdout.jsonl`).
- **Quality Floor**: Average quality score `>= 0.90` (evaluated by the Vertex Gen AI Evaluation Service autorater).
- **Architectural Rules**:
  - `MY_MODELS["coordinator"]` must be `"gemini-3.1-pro-preview"` or `"gemini-3.5-flash"` (cannot be `"gemini-3.1-flash-lite"`).
  - `MY_MODELS["embedding"]` must be `"gemini-embedding-001"`.

#### Winning Configuration
```python
MY_MODELS = {
    "coordinator": "gemini-3.5-flash",
    "docs":        "gemini-3.1-flash-lite",
    "community":   "gemini-3.1-flash-lite",
    "embedding":   "gemini-embedding-001",
}
MY_THINKING = "LOW"
MY_TOPK     = 2
MY_MAXOUT   = None  # Keeps the default 3,072 token ceiling without starving answers
```

#### Final Evaluation Results on Held-Out Set (16 Queries)
- **Cost per query**: **`$0.011415` USD** (Budget: `<= $0.0135` USD — **56% below baseline**)
- **Quality score**: **`0.9375`** (Quality Floor: `>= 0.90`)
- **Verdict**: **`PASS` (`"passed": true`)**

| Agent Component | Input Tokens | Thought Tokens | Output Tokens | Total Cost (16 Queries) |
| :--- | ---: | ---: | ---: | ---: |
| **Coordinator (`gemini-3.5-flash`)** | 17,359 | 4,824 | 9,340 | `$0.153514` |
| **DocsAgent (`gemini-3.1-flash-lite` + VAIS)** | 4,027 | 0 | 2,485 | `$0.021234` |
| **CommunityAgent (`gemini-3.1-flash-lite` + Embedding)** | 3,692 | 0 | 4,629 | `$0.007885` |
| **Total (16 Held-Out Queries)** | **25,078** | **4,824** | **16,454** | **`$0.182634` (`$0.011415` / query)** |

---

## 3. Reusable Code Patterns for Future Projects

### Pattern A: ADK Usage Ledger Callbacks (`code_assistant/agents.py` & `code_assistant/ledger.py`)
Attach `before_model_callback` and `after_model_callback` to every `google.adk.agents.Agent` to automatically record input, cached, output, and reasoning (`thoughts_token_count`) tokens per agent turn:

```python
from google.adk.agents import Agent
from google.genai import types as genai_types

def _make_callbacks(agent_name: str, model_id: str, ledger, qid_holder: dict):
    t0 = {}

    def before_model(callback_context, llm_request):
        t0[id(llm_request)] = time.perf_counter()
        return None

    def after_model(callback_context, llm_response):
        um = getattr(llm_response, "usage_metadata", None)
        if um is None:
            return None
        cached = int(getattr(um, "cached_content_token_count", 0) or 0)
        prompt = int(getattr(um, "prompt_token_count", 0) or 0)
        # prompt_token_count includes cached_content_token_count; subtract to avoid double-billing
        uncached_in = max(0, prompt - cached)
        out = int(getattr(um, "candidates_token_count", 0) or 0)
        thought = int(getattr(um, "thoughts_token_count", 0) or 0)
        ledger.add(
            query_id=qid_holder["v"],
            agent=agent_name,
            call_type="model",
            model=str(getattr(llm_response, "model_version", None) or model_id),
            input_tokens=uncached_in,
            cached_tokens=cached,
            output_tokens=out,
            thought_tokens=thought,
        )
        return None

    return before_model, after_model
```

### Pattern B: Controlling Gemini 3 Thinking Effort & Output Ceilings in ADK
Always bound `max_output_tokens` and explicitly set `thinking_level` (`"MINIMAL"`, `"LOW"`, `"MEDIUM"`, `"HIGH"`) so reasoning tokens do not silently inflate output costs:

```python
def _gen_config(max_output_tokens: int, thinking_level: str | None = None):
    tc = (
        genai_types.ThinkingConfig(
            thinking_level=getattr(genai_types.ThinkingLevel, thinking_level)
        )
        if thinking_level
        else None
    )
    return genai_types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        thinking_config=tc,
    )
```

### Pattern C: Context Caching (`caches.create`) for Large Static Prompts
When an agent's system instructions, reference manual, or few-shot examples exceed `2,048` tokens, cache the prefix once and pass `cached_content=cache.name` on subsequent calls for a 90% input token discount:

```python
from google import genai
from google.genai import types as genai_types

client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

cache = client.caches.create(
    model="gemini-3.5-flash",
    config=genai_types.CreateCachedContentConfig(
        system_instruction="You are a senior Python & Google Cloud code assistant.",
        contents=[LARGE_REFERENCE_MANUAL],
        ttl="600s",
    ),
)

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=user_question,
    config=genai_types.GenerateContentConfig(
        cached_content=cache.name,
        temperature=0.0,
        thinking_config=genai_types.ThinkingConfig(
            thinking_level=genai_types.ThinkingLevel.LOW
        ),
    ),
)
```

### Pattern D: Semantic Response Cache with Embeddings
Before invoking an expensive multi-agent workflow, check an exact-match hash first and then cosine similarity over `gemini-embedding-001` vectors:

```python
import hashlib
import numpy as np

class SemanticCache:
    def __init__(self, embed_fn, threshold=0.85):
        self.embed_fn = embed_fn
        self.threshold = threshold
        self.exact = {}   # sha256(normalized_query) -> answer
        self.entries = [] # list of (embedding_vec, query, answer)

    @staticmethod
    def _key(q: str) -> str:
        return hashlib.sha256(" ".join(q.lower().split()).encode()).hexdigest()

    def get(self, query: str):
        k = self._key(query)
        if k in self.exact:
            return self.exact[k], 1.0, None
        q_vec = np.asarray(self.embed_fn(query), dtype=float)
        best_sim, best_ans = -1.0, None
        for vec, _, ans in self.entries:
            sim = float(np.dot(q_vec, vec) / (np.linalg.norm(q_vec) * np.linalg.norm(vec)))
            if sim > best_sim:
                best_sim, best_ans = sim, ans
        if best_sim >= self.threshold:
            return best_ans, best_sim, q_vec
        return None, best_sim, q_vec

    def put(self, query: str, answer: str, q_vec=None):
        self.exact[self._key(query)] = answer
        if q_vec is None:
            q_vec = np.asarray(self.embed_fn(query), dtype=float)
        self.entries.append((q_vec, query, answer))
```

### Pattern E: Automated Quality Scoring with Vertex AI `EvalTask` (`code_assistant/quality.py`)
Use `PointwiseMetric` and `EvalTask` to score agent responses against reference answers on a 1–5 rubric normalized to `[0, 1]`:

```python
import pandas as pd
import vertexai
from vertexai.evaluation import EvalTask, PointwiseMetric, PointwiseMetricPromptTemplate

metric = PointwiseMetric(
    metric="code_assistant_answer_quality",
    metric_prompt_template=PointwiseMetricPromptTemplate(
        criteria={
            "factual_alignment": "Key technical facts, API names, flags, and fixes in reference appear in response.",
            "code_and_commands": "Code, CLI commands, and config snippets are syntactically valid and match reference.",
            "completeness": "Every part of a multi-part question is addressed with source citations.",
        },
        rating_rubric={
            "5": "Complete, accurate, actionable, and properly cited.",
            "4": "Covers main points accurately with minor omissions.",
            "3": "Partially correct; misses a key detail or one half of a two-part question.",
            "2": "Mostly incorrect, vague, or recommends deprecated APIs.",
            "1": "Irrelevant, empty, or fabricated.",
        },
        input_variables=["prompt", "reference"],
    ),
)

vertexai.init(project=PROJECT, location="us-central1")
eval_df = pd.DataFrame([
    {"prompt": r["query"], "response": r["answer"], "reference": r["reference"]}
    for r in answers
])
result = EvalTask(dataset=eval_df, metrics=[metric]).evaluate()
scores = pd.to_numeric(result.metrics_table["code_assistant_answer_quality/score"], errors="coerce").fillna(1.0)
normalized_quality = float(((scores.clip(1.0, 5.0) - 1.0) / 4.0).mean())
```

---

## 4. Repository Structure

```
.
├── README.md                                  # Consolidated engineering runbook & reusable code guide
├── code_assistant_cost_optimization.ipynb     # Completed lab notebook with Final Challenge solution
├── final_metrics.json                         # Verified PASS metrics uploaded to Cloud Storage
├── requirements.txt                           # Locked Python dependencies
├── code_assistant/
│   ├── __init__.py                            # Package exports
│   ├── agents.py                              # ADK Coordinator + Sub-Agents + Ledger callbacks
│   ├── ledger.py                              # UsageLedger for token, embedding & retrieval cost tracking
│   ├── retrieval.py                           # Vertex AI Embeddings + ChromaDB index & lookup tools
│   ├── vais.py                                # Discovery Engine (Agent Search) provisioning & search tool
│   ├── quality.py                             # Vertex Gen AI Evaluation Service wrapper & rubric
│   ├── display.py                             # HTML scorecard & verdict renderers
│   └── data.py                                # JSONL dataset loader
└── assets/
    ├── docs.jsonl                             # 20 official documentation entries for Agent Search
    ├── stackoverflow.jsonl                    # 20 Stack Overflow posts for ChromaDB
    ├── eval_dataset.jsonl                     # 15-query development evaluation dataset
    └── eval_holdout.jsonl                     # 16-query held-out final challenge dataset
```
