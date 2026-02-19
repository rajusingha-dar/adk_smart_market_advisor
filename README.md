# 🧠 Smart Market Advisor V2

A production-ready **multi-agent market research system** built on [Google ADK (Agent Development Kit)](https://google.github.io/adk-docs/). Given any business topic or industry, it runs a fully automated sequential research pipeline — market analysis, financial modeling, policy research, technology assessment, risk evaluation — and delivers a final business viability decision. All powered by GPT-3.5-Turbo via LiteLLM.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Agent Pipeline Deep Dive](#-agent-pipeline-deep-dive)
- [Token Optimization Strategy](#-token-optimization-strategy)
- [Logging & Observability](#-logging--observability)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the System](#-running-the-system)
- [Example Interaction](#-example-interaction)
- [Key Design Decisions](#-key-design-decisions)
- [Known Limitations & Roadmap](#-known-limitations--roadmap)

---

## 🔍 Overview

Smart Market Advisor V2 answers one question: **"Should I enter this market?"**

You provide a topic (e.g., *"EV market in India"*, *"AI software startups"*, *"solar panel manufacturing"*). The system automatically:

1. Researches market size, trends, and players
2. Analyzes financial viability — costs, margins, ROI
3. Scans government policy, subsidies, and regulations
4. Evaluates technology and infrastructure readiness
5. Identifies business, financial, and execution risks
6. Synthesizes everything into a **Go / No-Go decision** with clear reasoning

Every research stage is compressed before being passed to the next agent, preventing context window bloat and minimizing token costs.

---

## 🏗 System Architecture

```
User Input
    │
    ▼
┌─────────────────┐
│   router_agent  │  ← Entry point. Handles greetings, routes research
│   (LlmAgent)   │    queries, asks clarifying questions.
└────────┬────────┘
         │ delegates to
         ▼
┌──────────────────────────────────────────────────────────┐
│              market_research_pipeline                    │
│                  (SequentialAgent)                       │
│                                                          │
│  market_agent → market_summarizer                        │
│       ↓                ↓                                 │
│  finance_agent → finance_summarizer                      │
│       ↓                ↓                                 │
│  policy_agent  → policy_summarizer                       │
│       ↓                ↓                                 │
│  tech_agent    → tech_summarizer                         │
│       ↓                ↓                                 │
│  risk_agent    → risk_summarizer                         │
│       ↓                                                  │
│  decision_agent  ← reads all *_summary keys from state  │
└──────────────────────────────────────────────────────────┘
```

**Key pattern:** Every research agent writes a `*_raw` key into ADK session state. Its paired summarizer immediately compresses that output to ~150-180 tokens and writes a `*_summary` key. This keeps the context window lean across 11 sequential agent calls.

---

## 📁 Project Structure

```
smart-market-advisor-v2/
│
├── root_agent/                     # Main agent package (ADK convention)
│   ├── __init__.py
│   ├── agent.py                    # Exports root_agent (entry point for ADK)
│   ├── router.py                   # router_agent definition
│   ├── pipeline.py                 # SequentialAgent pipeline assembly
│   ├── callbacks.py                # before_model callback (context trimming)
│   │
│   ├── sub_agents/                 # One file per agent
│   │   ├── market_agent.py         # Market size, trends, players
│   │   ├── market_summarizer.py    # Compresses market_raw → market_summary
│   │   ├── finance_agent.py        # Costs, margins, ROI
│   │   ├── finance_summarizer.py   # Compresses finance_raw → finance_summary
│   │   ├── policy_agent.py         # Government incentives, regulations
│   │   ├── policy_summarizer.py    # Compresses policy_raw → policy_summary
│   │   ├── tech_agent.py           # Technology & infrastructure trends
│   │   ├── tech_summarizer.py      # Compresses tech_raw → tech_summary
│   │   ├── risk_agent.py           # Business, market, regulatory risks
│   │   ├── risk_summarizer.py      # Compresses risk_raw → risk_summary
│   │   ├── decision_agent.py       # Final Go/No-Go verdict
│   │   └── summarizer_agent.py     # (Legacy, not in active pipeline)
│   │
│   ├── tools/
│   │   └── web_search.py           # Tavily API search tool (FunctionTool)
│   │
│   ├── memory/
│   │   ├── sliding_context.py      # Trims conversation to last 6 messages
│   │   ├── memory_manager.py       # Manages SlidingContext
│   │   ├── summarizer_gate.py      # (Legacy factory, not in active pipeline)
│   │   └── context_pruner.py       # (Legacy pruner, not in active pipeline)
│   │
│   └── logging/
│       ├── logger.py               # Appends JSONL events to token_log.jsonl
│       ├── token_meter.py          # tiktoken-based token counter
│       └── state_logger.py         # Per-session state snapshots to disk
│
├── config/                         # (Reserved for future YAML agent configs)
├── main.py                         # Placeholder entry point
├── run_server.py                   # Launches ADK web UI
├── pyproject.toml                  # Project dependencies (uv/pip)
├── .python-version                 # Python 3.12
└── token_log.jsonl                 # Auto-generated token usage log
```

---

## 🔬 Agent Pipeline Deep Dive

### 1. `router_agent`
**File:** `router.py`

The user-facing entry point. Uses GPT-3.5-Turbo. Handles three cases:
- **Greeting** → Responds warmly and asks what topic to research
- **Research query** → Delegates to `market_research_pipeline`
- **Unclear input** → Asks a clarifying question

```python
router_agent = Agent(
    name="router_agent",
    model=LiteLlm(model="gpt-3.5-turbo"),
    sub_agents=[market_research_pipeline],
    ...
)
```

---

### 2. `market_agent` + `market_summarizer`
**Files:** `sub_agents/market_agent.py`, `sub_agents/market_summarizer.py`

**market_agent** uses `web_search` to research:
- Market size and growth rate
- Key players and demand drivers
- Emerging trends

Writes output to ADK state key: `market_raw`

**market_summarizer** reads `market_raw`, compresses it to ≤180 tokens (facts and numbers only), logs the raw token count, and writes to `market_summary`.

---

### 3. `finance_agent` + `finance_summarizer`
**Files:** `sub_agents/finance_agent.py`, `sub_agents/finance_summarizer.py`

**finance_agent** uses `web_search` to research:
- Cost structure and startup costs
- Revenue models and pricing
- Margins and ROI projections

Writes output to state key: `finance_raw`

**finance_summarizer** compresses to `finance_summary` (≤180 tokens).

---

### 4. `policy_agent` + `policy_summarizer`
**Files:** `sub_agents/policy_agent.py`, `sub_agents/policy_summarizer.py`

**policy_agent** researches:
- Government subsidies and incentives
- Compliance requirements and regulations
- National and state-level programs
- Policy change risks

Writes to: `policy_raw` → compressed to `policy_summary`

---

### 5. `tech_agent` + `tech_summarizer`
**Files:** `sub_agents/tech_agent.py`, `sub_agents/tech_summarizer.py`

**tech_agent** researches:
- Infrastructure readiness
- Battery/core technology maturity
- Manufacturing process innovations
- Cost reduction trends

Writes to: `tech_raw` → compressed to `tech_summary`

---

### 6. `risk_agent` + `risk_summarizer`
**Files:** `sub_agents/risk_agent.py`, `sub_agents/risk_summarizer.py`

**risk_agent** identifies:
- Market risks (competition, saturation)
- Financial risks (capital requirements, payback period)
- Regulatory risks (policy reversals, compliance gaps)
- Technology and execution risks

Writes to: `risk_raw` → compressed to `risk_summary` (≤150 tokens)

---

### 7. `decision_agent`
**File:** `sub_agents/decision_agent.py`

The final synthesis agent. Reads all `*_summary` keys from session state and produces:

```
Verdict: Yes / No / Maybe
Top 3 Reasons: ...
Next Steps: ...
```

Writes output to state key: `decision`.

---

## ⚡ Token Optimization Strategy

Context window bloat is the biggest cost problem in long sequential pipelines. This system uses a **Compress-As-You-Go** pattern:

```
Agent Output (400-600 tokens)
         ↓
    Summarizer
         ↓
Compressed Summary (≤180 tokens)
         ↓
  Next Agent receives clean, lean context
```

**Why this matters (from real token_log.jsonl data):**

| Stage | Raw Tokens (typical) | Target Summary |
|-------|---------------------|----------------|
| Market | 400–530 tokens | ≤180 tokens |
| Finance | 448–524 tokens | ≤180 tokens |
| Policy | 357–436 tokens | ≤180 tokens |
| Tech | 7–385 tokens | ≤180 tokens |
| Risk | 357–603 tokens | ≤150 tokens |

Without compression, by the time `decision_agent` runs, the context would contain 2,000–2,800 tokens of raw research. With compression, it receives ~900 tokens of clean summaries — a **55–65% reduction**.

### Dynamic Instruction Pattern

Each summarizer uses a **callable instruction** instead of a static string:

```python
def summarize_instruction(ctx):
    raw = ctx.state.get("market_raw", "")
    raw_tokens = count_tokens(raw)        # Count before compression
    log_event("token_input", {...})       # Log for observability
    return f"Compress this to ≤180 tokens:\n{raw}"

market_summarizer = Agent(
    instruction=summarize_instruction,   # ADK calls this at runtime
    output_key="market_summary",
    ...
)
```

This pattern lets the summarizer inject the actual raw content into its own prompt at runtime — no hardcoded text, always fresh from state.

---

## 📊 Logging & Observability

### `token_log.jsonl`
Auto-generated file. Every web search and every summarizer stage logs an event:

```jsonl
{"event": "web_search_request", "query": "EV market outlook in India 2025", "ts": 1770130785.38}
{"event": "web_search_response", "query": "EV market outlook in India 2025", "results_count": 5, "ts": 1770130789.83}
{"event": "token_input", "stage": "market", "raw_tokens": 525, "ts": 1770131440.35}
{"event": "token_input", "stage": "finance", "raw_tokens": 448, "ts": 1770131447.79}
```

Use this to monitor cost patterns, identify which stages are most verbose, and tune your summarizer token targets.

### `state_logs/`
If `state_logger.py` is wired in, per-session state snapshots are written as JSONL to `state_logs/{session_id}.jsonl`. Useful for debugging what each agent stored.

### Token Counter
`logging/token_meter.py` uses `tiktoken` (cl100k_base encoding — same as GPT-3.5/GPT-4) to count tokens accurately before any LLM call:

```python
from root_agent.logging.token_meter import count_tokens
count_tokens("some text")  # Returns int
```

---

## ✅ Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.12+ |
| Google ADK | ≥1.23.0 |
| LiteLLM | ≥1.81.6 |
| OpenAI API Key | Required (via LiteLLM) |
| Tavily API Key | Required (web search) |

---

## 🛠 Installation

### Option 1: Using `uv` (recommended — fast)

```bash
# Install uv if you don't have it
pip install uv

# Clone the repo
git clone https://github.com/your-username/smart-market-advisor-v2.git
cd smart-market-advisor-v2

# Install dependencies
uv sync
```

### Option 2: Using `pip`

```bash
git clone https://github.com/your-username/smart-market-advisor-v2.git
cd smart-market-advisor-v2

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install google-adk litellm tiktoken python-dotenv requests
```

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# OpenAI API key (used by LiteLLM to power all agents)
OPENAI_API_KEY=sk-...

# Tavily API key (used by web_search tool)
TAVILY_API_KEY=tvly-...
```

> **Get your keys:**
> - OpenAI: https://platform.openai.com/api-keys
> - Tavily: https://app.tavily.com

ADK picks up `.env` automatically. You can also export them as shell environment variables.

---

## 🚀 Running the System

### ADK Web UI (recommended for development)

```bash
adk web
```

Then open [http://localhost:8000](http://localhost:8000) in your browser. Select `root_agent` from the dropdown and start chatting.

### Alternatively via `run_server.py`

```bash
python run_server.py
```

### ADK CLI (for quick terminal testing)

```bash
adk run root_agent
```

---

## 💬 Example Interaction

```
You:    Hi

Agent:  Hello! I can help with market research and business feasibility 
        analysis on any topic or industry. What would you like to explore?

You:    EV market in India

Agent:  [Runs full pipeline: market → finance → policy → tech → risk → decision]

        === BUSINESS DECISION ===
        Verdict: YES

        Top 3 Reasons:
        1. India's EV market is growing at 49% CAGR with strong government backing
        2. Policy subsidies (FAME II, PLI scheme) significantly reduce entry costs
        3. Battery costs dropping 15–20% annually improve margin outlook

        Next Steps:
        1. Partner with an established charging infrastructure provider
        2. Target Tier-1 cities first (Delhi, Bengaluru, Mumbai) for fastest adoption
        3. Apply for PLI scheme benefits before next fiscal year cutoff
```

---

## 🎯 Key Design Decisions

### Why SequentialAgent instead of ParallelAgent?

Research stages build on each other — finance analysis benefits from knowing market size, and decision synthesis needs all five domains. Sequential ordering ensures each agent has relevant context from prior stages in ADK's session state.

### Why GPT-3.5-Turbo for all agents?

Cost and speed. Each full pipeline run makes ~11 LLM calls + 1–5 web searches. Using GPT-3.5-Turbo keeps costs low (roughly $0.002–0.005 per full research run) while still producing quality output for structured summarization tasks.

### Why callable instructions for summarizers?

ADK evaluates the `instruction` parameter at agent invocation time if it's a callable. This lets summarizers dynamically read from session state (`ctx.state.get("market_raw")`) and embed the actual content directly into the prompt — cleaner than relying on ADK's automatic state injection for large text blocks.

### Why Tavily over Google Search?

Tavily returns pre-parsed, clean text content from search results — no HTML scraping needed. The `search_depth="advanced"` setting returns richer, more structured content suitable for research tasks.

---

## ⚠️ Known Limitations & Roadmap

**Current limitations:**

- `decision_agent` instruction references `market_raw`/`finance_raw` in comments but these keys are already replaced by summaries by the time it runs — verify the prompt references `*_summary` keys
- `callbacks.py` (`trim_context`) is defined but not wired to any agent's `before_model_callback` yet
- `summarizer_agent.py` and `summarizer_gate.py` are legacy files not used in the active pipeline — safe to remove

**Potential improvements:**

- [ ] Upgrade `decision_agent` to GPT-4 for higher quality synthesis while keeping research agents on GPT-3.5-Turbo
- [ ] Add `after_model_callback` to each summarizer to log compression ratios (raw vs summary tokens)
- [ ] Wire `DatabaseSessionService` for conversation persistence across sessions
- [ ] Add `ParallelAgent` to run market + finance + policy + tech + risk simultaneously, then feed all summaries to decision_agent
- [ ] Add a `report_agent` that formats the final output as a structured PDF or Word document
- [ ] Build a simple Streamlit or FastAPI frontend as an alternative to ADK web UI

---

## 📄 License

MIT License. See `LICENSE` for details.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/parallel-research`)
3. Commit your changes
4. Push and open a PR