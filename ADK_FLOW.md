# 🧩 ADK Code Flow & Framework Learning Guide
### Smart Market Advisor V2 — A Deep-Dive Reference

This document is a complete walkthrough of how Google's **Agent Development Kit (ADK)** is used in this project — tracing every line of execution, explaining every important argument, import, function, and ADK concept. It is written as a learning key for developers new to ADK.

---

## 📚 Table of Contents

1. [ADK Mental Model — The Big Picture](#1-adk-mental-model--the-big-picture)
2. [Entry Point — How ADK Discovers Your Agent](#2-entry-point--how-adk-discovers-your-agent)
3. [The Agent Class — Core Building Block](#3-the-agent-class--core-building-block)
4. [LiteLlm — Connecting Non-Google Models](#4-litellm--connecting-non-google-models)
5. [The `instruction` Argument — Static vs Dynamic](#5-the-instruction-argument--static-vs-dynamic)
6. [The `output_key` Argument — Writing to Session State](#6-the-output_key-argument--writing-to-session-state)
7. [Session State — The Shared Memory Between Agents](#7-session-state--the-shared-memory-between-agents)
8. [The `sub_agents` Argument — Building Agent Trees](#8-the-sub_agents-argument--building-agent-trees)
9. [SequentialAgent — Ordered Pipeline Execution](#9-sequentialagent--ordered-pipeline-execution)
10. [FunctionTool — Giving Agents Custom Tools](#10-functiontool--giving-agents-custom-tools)
11. [Callbacks — Intercepting Agent Lifecycle Events](#11-callbacks--intercepting-agent-lifecycle-events)
12. [CallbackContext & ReadonlyContext](#12-callbackcontext--readonlycontext)
13. [The Full Execution Flow — Step by Step](#13-the-full-execution-flow--step-by-step)
14. [State Scopes — session vs user vs app vs temp](#14-state-scopes--session-vs-user-vs-app-vs-temp)
15. [The Dynamic Instruction Pattern — This Project's Key Technique](#15-the-dynamic-instruction-pattern--this-projects-key-technique)
16. [Complete Import Reference](#16-complete-import-reference)
17. [Common Pitfalls & ADK Gotchas](#17-common-pitfalls--adk-gotchas)

---

## 1. ADK Mental Model — The Big Picture

Before reading a single line of code, understand this core mental model:

```
ADK Application
│
├── Agent Tree (nested agents)
│     ├── root_agent          ← What ADK exposes to the user
│     │     └── sub_agents    ← Agents this one can delegate to
│     │           └── ...
│
├── Session                  ← A single conversation
│     └── State (dict)       ← Shared memory between ALL agents in this session
│
└── InvocationContext        ← Runtime object passed to every agent/callback
      ├── session
      ├── agent
      └── invocation_id
```

**Key insight:** ADK agents don't pass data to each other by return values or function arguments. They communicate through **session state** — a shared dictionary that every agent in the session can read and write. `output_key` is the mechanism that writes to this dictionary automatically.

---

## 2. Entry Point — How ADK Discovers Your Agent

**File:** `root_agent/agent.py`

```python
from .router import router_agent

root_agent = router_agent
```

**This is the single most important convention in ADK.**

When you run `adk web` or `adk run`, ADK scans the directory for a Python package and looks for a module-level variable named **exactly `root_agent`**. This is what ADK registers as the entry point for the entire application.

```
adk web
  ↓
Scans for root_agent/agent.py  (or agent.py in root)
  ↓
Finds: root_agent = router_agent
  ↓
Registers router_agent as the user-facing entry point
```

**Why a separate `agent.py`?**  
ADK's convention is that `agent.py` is a thin re-export file. All the actual agent definitions live elsewhere (`router.py`, `pipeline.py`, etc.) and get imported here. This keeps the entry point file clean and makes it easy to swap the root agent without changing business logic files.

**What happens if `root_agent` is missing?**  
ADK raises an error. The variable name must be exactly `root_agent` — ADK looks for this by name.

---

## 3. The Agent Class — Core Building Block

**File:** `root_agent/router.py`, all `sub_agents/*.py`

**Import:**
```python
from google.adk.agents import Agent
```

`Agent` is an alias for `LlmAgent` — the primary agent class in ADK. Every specialist agent in this project (`market_agent`, `finance_agent`, etc.) is an instance of this class.

### Full Signature with All Arguments Used in This Project

```python
from google.adk.agents import Agent
from google.adk.models import LiteLlm

agent = Agent(
    # ── REQUIRED ──────────────────────────────────────────────────────
    name="agent_name",
    # Must be a valid Python identifier. Must be UNIQUE across the entire
    # agent tree. ADK uses this name to route transfers and track state.
    # Cannot be "user" (reserved).

    # ── MODEL ─────────────────────────────────────────────────────────
    model=LiteLlm(model="gpt-3.5-turbo"),
    # The LLM to use. Accepts a string (for Gemini models) or a BaseLlm
    # object (for non-Gemini models via LiteLlm). If not set, inherits
    # from parent agent. Default: "gemini-2.5-flash".

    # ── IDENTITY ──────────────────────────────────────────────────────
    description="One-line description of what this agent does.",
    # CRITICAL for multi-agent routing. When an LLM-based agent decides
    # which sub-agent to delegate to, it reads each sub-agent's description
    # to choose. Make this precise and specific. One line is preferred.

    instruction="You are a ...",
    # The system prompt for this agent. Tells the LLM what role to play,
    # what to do, and what rules to follow. Can be a string or a callable
    # (function) that returns a string — see Section 5 for the dynamic
    # instruction pattern.

    # ── OUTPUT ────────────────────────────────────────────────────────
    output_key="state_key_name",
    # When set, ADK automatically saves this agent's final text response
    # into session state under this key. This is the primary mechanism
    # for agents to share data with each other. See Section 6.

    # ── TOOLS ─────────────────────────────────────────────────────────
    tools=[web_search],
    # List of tools this agent can call. Can be FunctionTool instances,
    # plain Python functions, or ADK built-in toolsets.

    # ── DELEGATION ────────────────────────────────────────────────────
    sub_agents=[pipeline],
    # Agents that this agent can delegate tasks to. The LLM decides when
    # to transfer control based on sub-agent descriptions. See Section 8.
)
```

### What Each Argument Does Internally

| Argument | Type | Internal Behavior |
|----------|------|-------------------|
| `name` | `str` | Used as author ID in events; must be unique in agent tree |
| `model` | `str \| BaseLlm` | Inherited by children if not set; default is `gemini-2.5-flash` |
| `description` | `str` | Read by parent LLM to decide when to delegate |
| `instruction` | `str \| Callable` | Becomes the system prompt; evaluated at runtime if callable |
| `output_key` | `str \| None` | Triggers `__maybe_save_output_to_state` after final response |
| `tools` | `list` | Converted to `BaseTool` instances via `canonical_tools()` |
| `sub_agents` | `list[BaseAgent]` | Registered in agent tree; parent is set automatically |

---

## 4. LiteLlm — Connecting Non-Google Models

**File:** All agent files

**Import:**
```python
from google.adk.models import LiteLlm
```

ADK is built around Google's Gemini models. To use OpenAI, Anthropic, or any other model, ADK provides a `LiteLlm` wrapper that acts as a bridge using the [LiteLLM](https://docs.litellm.ai/) library.

```python
MODEL = LiteLlm(model="gpt-3.5-turbo")
```

**How it works:**
```
Agent calls model
  ↓
ADK checks: is model a BaseLlm instance?
  ↓ Yes
LiteLlm.generate_content_async() is called
  ↓
LiteLlm translates the Gemini-format request → OpenAI format
  ↓
Calls OpenAI API using OPENAI_API_KEY from environment
  ↓
Translates OpenAI response → Gemini-format response
  ↓
ADK processes the response normally
```

**Why `MODEL = LiteLlm(...)` is defined as a module-level constant:**  
It's instantiated once at import time and reused across all agents. Avoids creating multiple identical objects. If you change the model, you change it in one place.

**Model string format:**  
LiteLlm uses the same model string format as the LiteLLM library: `"gpt-3.5-turbo"`, `"gpt-4"`, `"claude-3-sonnet-20240229"`, etc.

**Required environment variable:**
```bash
OPENAI_API_KEY=sk-...    # For OpenAI models
```

---

## 5. The `instruction` Argument — Static vs Dynamic

**Files:** `router.py` (static), `sub_agents/market_summarizer.py` (dynamic)

This is one of the most powerful and least understood features of ADK.

### Static Instruction (String)

```python
# router.py
router_agent = Agent(
    instruction="""
You are the main router.
...
""",
)
```

When `instruction` is a plain string, ADK sends it as the system prompt verbatim. It's fixed at agent definition time and never changes.

### Dynamic Instruction (Callable)

```python
# market_summarizer.py
def summarize_instruction(ctx):           # ctx is ReadonlyContext
    raw = ctx.state.get("market_raw", "") # Read from session state
    raw_tokens = count_tokens(raw)        # Count tokens before compression

    instruction = f"""
You compress research output.
...
TEXT:
{raw}
"""
    log_event("token_input", {"stage": "market", "raw_tokens": raw_tokens})
    return instruction                    # Return the constructed string

market_summarizer = Agent(
    instruction=summarize_instruction,   # Pass the FUNCTION, not a call to it
    output_key="market_summary",
)
```

**What ADK does with a callable instruction:**

```
Agent is invoked
  ↓
ADK checks: is instruction a string or callable?
  ↓ Callable
ADK calls: instruction(ctx)  where ctx = ReadonlyContext(invocation_context)
  ↓
Function reads ctx.state to get "market_raw"
  ↓
Function builds and returns the full prompt string with raw content embedded
  ↓
ADK uses that returned string as the system prompt for this LLM call
```

**Why this is powerful:**  
The summarizer dynamically injects the content it needs to compress directly into its own system prompt at runtime. The alternative — relying on ADK to inject state variables into prompt templates via `{variable_name}` syntax — also works, but the callable approach gives you full Python control: you can count tokens, log before the call, do conditional logic, and format the content however you need.

**The `ctx` parameter:**  
It is a `ReadonlyContext` instance. It gives read-only access to:
- `ctx.state` → The current session state (as a `MappingProxyType`)
- `ctx.user_content` → The message that triggered this invocation
- `ctx.invocation_id` → Unique ID for this run
- `ctx.session` → The full `Session` object
- `ctx.user_id` → The user's ID
- `ctx.agent_name` → The name of the agent being run

---

## 6. The `output_key` Argument — Writing to Session State

**Files:** `market_agent.py`, `market_summarizer.py`, and all research/summarizer agents

```python
market_agent = Agent(
    ...
    output_key="market_raw",
)
```

### What ADK does with `output_key`

After the agent finishes generating its final response, ADK internally calls `__maybe_save_output_to_state()`:

```python
# Inside LlmAgent (ADK source)
def __maybe_save_output_to_state(self, event: Event):
    if (
        self.output_key
        and event.is_final_response()   # Only on the FINAL response, not partials
        and event.content
        and event.content.parts
    ):
        result = ''.join(
            part.text for part in event.content.parts
            if part.text and not part.thought
        )
        event.actions.state_delta[self.output_key] = result
        #                                  ↑
        # Stored in state_delta, not directly in session.state yet.
        # The session service commits the delta to session.state
        # when it calls append_event().
```

**Key insight: state_delta vs session.state**

State is not written directly. ADK uses a `state_delta` pattern:
1. Agent generates response
2. ADK puts `output_key → response_text` into `event.actions.state_delta`
3. When the session service calls `append_event()`, the delta is merged into `session.state`
4. All subsequent agents in the same session see the updated state

This is why in `market_summarizer.py`, `ctx.state.get("market_raw", "")` works — by the time the summarizer runs, `market_agent` has already committed its output to state via this delta mechanism.

### State Key Naming in This Project

| Agent | `output_key` | Contains |
|-------|-------------|----------|
| `market_agent` | `"market_raw"` | Full market research text (400–530 tokens) |
| `market_summarizer` | `"market_summary"` | Compressed market facts (≤180 tokens) |
| `finance_agent` | `"finance_raw"` | Full financial analysis |
| `finance_summarizer` | `"finance_summary"` | Compressed financial data |
| `policy_agent` | `"policy_raw"` | Full policy research |
| `policy_summarizer` | `"policy_summary"` | Compressed policy data |
| `tech_agent` | `"tech_raw"` | Full technology research |
| `tech_summarizer` | `"tech_summary"` | Compressed tech data |
| `risk_agent` | `"risk_raw"` | Full risk analysis |
| `risk_summarizer` | `"risk_summary"` | Compressed risk data |
| `decision_agent` | `"decision"` | Final Go/No-Go verdict |

---

## 7. Session State — The Shared Memory Between Agents

**ADK Source:** `sessions/state.py`

Session state is the backbone of agent-to-agent communication in ADK. It is a Python dictionary that persists for the lifetime of a session and is accessible to every agent in that session.

### The State Object

ADK wraps the session state dictionary in a `State` class that maintains both the current value and a pending delta:

```python
class State:
    def __init__(self, value: dict, delta: dict):
        self._value = value   # Committed state
        self._delta = delta   # Pending, uncommitted changes

    def __getitem__(self, key):
        if key in self._delta:     # Delta takes priority
            return self._delta[key]
        return self._value[key]

    def __setitem__(self, key, value):
        self._value[key] = value   # Updates both
        self._delta[key] = value
```

### How to Read State (in instruction callable)

```python
def summarize_instruction(ctx):
    # ctx is ReadonlyContext — state is a MappingProxyType (immutable)
    raw = ctx.state.get("market_raw", "")
    #              ↑ key name          ↑ default if missing
```

### How to Write State (in callbacks)

```python
def my_callback(callback_context):
    # callback_context is CallbackContext — state is writable
    callback_context.state["my_key"] = "my_value"
    # This goes into state_delta and is committed on append_event()
```

### How `output_key` Writes State (automatic)

```python
# You don't need to write state manually if you use output_key:
market_agent = Agent(output_key="market_raw")
# ADK automatically writes agent's final text to state["market_raw"]
```

---

## 8. The `sub_agents` Argument — Building Agent Trees

**File:** `router.py`

```python
router_agent = Agent(
    name="router_agent",
    ...
    sub_agents=[market_research_pipeline],
)
```

### What `sub_agents` Does

When you add agents to `sub_agents`, ADK:
1. Registers them in the agent tree (`parent_agent` is set automatically)
2. Makes them available for the LLM to delegate to
3. Gives the LLM a hidden `transfer_to_agent` tool

When `router_agent`'s LLM decides to route a market query, it calls `transfer_to_agent("market_research_pipeline")` internally. ADK intercepts this function call and runs the pipeline.

```
User sends: "Tell me about the EV market in India"
  ↓
router_agent LLM sees instruction: "route to market_research_pipeline for market queries"
  ↓
LLM generates: transfer_to_agent("market_research_pipeline")
  ↓
ADK intercepts this function call
  ↓
ADK runs market_research_pipeline
  ↓
Pipeline output is returned as the final response
```

### How ADK Decides Which Sub-Agent to Use

The LLM reads each sub-agent's `description` field. This is why descriptions are critical:

```python
# Bad description — too vague
market_research_pipeline = SequentialAgent(
    description="Does stuff",
)

# Good description — tells the router exactly when to use it
market_research_pipeline = SequentialAgent(
    description="Market → Finance → Policy → Tech → Risk → Decision analysis pipeline",
)
```

### Important Constraint

An agent can only have **one parent** — you cannot add the same agent instance to two different agent trees. If you need the same logic in two places, create two instances.

---

## 9. SequentialAgent — Ordered Pipeline Execution

**File:** `pipeline.py`

**Import:**
```python
from google.adk.agents import SequentialAgent
```

`SequentialAgent` is a **workflow agent** — it does not have its own LLM. It runs its `sub_agents` one after another in the order listed.

```python
market_research_pipeline = SequentialAgent(
    name="market_research_pipeline",
    description="Market → Compress → Finance → Compress → ... → Decision",
    sub_agents=[
        market_agent,        # Step 1
        market_summarizer,   # Step 2
        finance_agent,       # Step 3
        finance_summarizer,  # Step 4
        policy_agent,        # Step 5
        policy_summarizer,   # Step 6
        tech_agent,          # Step 7
        tech_summarizer,     # Step 8
        risk_agent,          # Step 9
        risk_summarizer,     # Step 10
        decision_agent,      # Step 11
    ],
)
```

### How SequentialAgent Works Internally

```python
# Simplified from ADK's sequential_agent.py
async def _run_async_impl(self, ctx):
    for sub_agent in self.sub_agents:
        async for event in sub_agent.run_async(ctx):
            yield event
        # After each sub_agent finishes, the SAME ctx (with updated state)
        # is passed to the next sub_agent
```

**Critical point:** All sub-agents share the **same `InvocationContext`** (and therefore the same `session.state`). When `market_agent` writes to `state["market_raw"]`, `market_summarizer` immediately sees it because they use the same session.

### Sequential vs Parallel vs Loop

| Agent Type | When to Use |
|-----------|-------------|
| `SequentialAgent` | Steps depend on previous output (pipeline) |
| `ParallelAgent` | Steps are independent (can run simultaneously) |
| `LoopAgent` | Repeat until a condition is met |
| `Agent` (LlmAgent) | Any step requiring LLM reasoning |

### Why Sequential Over Parallel Here?

This project uses Sequential because:
- `finance_agent` benefits from knowing market size (found by `market_agent`)
- `decision_agent` needs all 5 summaries already written to state
- Each summarizer must run immediately after its research agent

If you ran all 5 research agents in parallel (with `ParallelAgent`), they would all share state but their order of completion would be non-deterministic.

---

## 10. FunctionTool — Giving Agents Custom Tools

**File:** `tools/web_search.py`

**Import:**
```python
from google.adk.tools.function_tool import FunctionTool
```

`FunctionTool` wraps a Python function and exposes it as a tool that an LLM agent can call. ADK uses the function's **signature and docstring** to generate the tool schema that the LLM sees.

```python
def _web_search(query: str) -> str:
    """Search the web for a given query."""    # LLM reads this docstring
    api_key = os.getenv("TAVILY_API_KEY")
    ...
    return text or "No results found."

web_search = FunctionTool(_web_search)
#                         ↑
# Pass the function object (no parentheses — don't call it)
```

### How the LLM Uses the Tool

```
market_agent instruction says: "If data is missing, use web_search"
  ↓
market_agent LLM decides to search
  ↓
LLM generates: {"function_call": {"name": "_web_search", "args": {"query": "EV market India 2025"}}}
  ↓
ADK intercepts this function call
  ↓
ADK calls: _web_search(query="EV market India 2025")
  ↓
Returns: "India's EV market grew 49% in 2024..."
  ↓
ADK feeds result back to the LLM as a tool_result
  ↓
LLM continues generating with the search result in context
```

### Tool Schema Generation

ADK automatically generates the JSON schema for your tool from the Python function:

```python
def _web_search(query: str) -> str:
#               ↑ type hint → schema: {"type": "string"}
#                              ↑ return type hint (informational)
```

**Best practices for FunctionTool:**
- Always use type hints — ADK uses them for schema generation
- The docstring becomes the tool description the LLM sees
- Return type should be `str` (ADK expects string tool results for LLM)
- Keep the function name meaningful — the LLM sees it

### Alternative: Plain Functions as Tools

ADK also accepts plain Python functions directly in the `tools` list without wrapping in `FunctionTool`. When ADK processes the tools list, it wraps bare callables in `FunctionTool` automatically:

```python
# Both of these work:
tools=[FunctionTool(_web_search)]   # Explicit
tools=[_web_search]                  # ADK wraps it automatically
```

In this project, `FunctionTool` is used explicitly — which is clearer and allows pre-configuration.

---

## 11. Callbacks — Intercepting Agent Lifecycle Events

**File:** `callbacks.py`

ADK provides hooks at multiple points in an agent's execution lifecycle. Callbacks let you intercept, inspect, and modify behavior without changing the agent itself.

### The Callback Lifecycle

```
User sends message
      │
      ▼
[before_agent_callback]     ← Runs before the agent starts
      │
      ▼
   Agent runs
      │
      ├─── [before_model_callback]  ← Runs before each LLM call
      │         │
      │         ▼
      │    LLM generates
      │         │
      │    [after_model_callback]   ← Runs after each LLM call
      │         │
      │         ▼
      │    If tool call:
      │    [before_tool_callback]   ← Runs before each tool call
      │         │
      │    Tool executes
      │         │
      │    [after_tool_callback]    ← Runs after each tool call
      │
      ▼
[after_agent_callback]      ← Runs after the agent finishes
```

### The `before_model` Callback in This Project

```python
# callbacks.py
from .memory.sliding_context import trim_context

def before_model(ctx):
    ctx.messages = trim_context(ctx.messages)
    #  ↑ ctx here is CallbackContext, but for before_model_callback
    #    it also receives llm_request which has the messages list
```

**Note:** This callback is defined but not yet wired to any agent via `before_model_callback=before_model`. To activate it:

```python
market_agent = Agent(
    ...
    before_model_callback=before_model,   # Wire it here
)
```

### Callback Signatures

```python
# before_agent_callback / after_agent_callback
def my_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    # Return None to continue normally
    # Return Content to skip the agent and use this response instead

# before_model_callback
def my_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    # llm_request.messages contains the full message list
    # Mutate llm_request to modify the request
    # Return None to continue normally
    # Return LlmResponse to skip the LLM call entirely

# after_model_callback
def my_after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    # Return None to use original response
    # Return LlmResponse to replace it

# before_tool_callback
def my_tool_callback(tool, args: dict, tool_context: ToolContext) -> Optional[dict]:
    # Return None to let tool execute
    # Return dict to skip tool and use this as result

# after_tool_callback
def my_after_tool_callback(tool, args: dict, tool_context: ToolContext, tool_response: dict) -> Optional[dict]:
    # Return None to use original response
    # Return dict to replace it
```

### The `trim_context` Function

```python
# sliding_context.py
MAX_MESSAGES = 6

def trim_context(messages):
    system = [m for m in messages if m["role"] == "system"]
    recent = [m for m in messages if m["role"] != "system"][-MAX_MESSAGES:]
    return system + recent
```

This implements a **sliding window** over the conversation history. ADK accumulates all messages in the session. As conversations grow long, the context window fills up. This function keeps only the system prompt and the last 6 non-system messages, preventing context overflow.

**Why preserve system messages?** System messages contain the agent's instruction (its personality and rules). They must always be included.

---

## 12. CallbackContext & ReadonlyContext

**ADK Source:** `agents/callback_context.py`, `agents/readonly_context.py`

These two classes are what you receive as `ctx` in different parts of ADK.

### ReadonlyContext

Received in: **callable instructions** (`instruction=my_func`)

```python
class ReadonlyContext:
    @property
    def state(self) -> MappingProxyType:   # READ-ONLY dict
    @property
    def user_content(self) -> types.Content
    @property
    def invocation_id(self) -> str
    @property
    def agent_name(self) -> str
    @property
    def session(self) -> Session
    @property
    def user_id(self) -> str
    @property
    def run_config(self) -> RunConfig
```

`state` here is a `MappingProxyType` — a read-only view of the session state. You can read but not write.

### CallbackContext

Received in: **all callback functions** (`before_agent_callback`, `before_model_callback`, etc.)

```python
class CallbackContext(ReadonlyContext):
    @property
    def state(self) -> State:   # WRITABLE State object (overrides ReadonlyContext.state)

    async def load_artifact(filename, version) -> types.Part
    async def save_artifact(filename, artifact) -> int
    async def list_artifacts() -> list[str]
    async def add_session_to_memory() -> None
```

`state` here is a writable `State` object. You can do `callback_context.state["my_key"] = "value"` and it will be committed to session state.

**The key difference:**

```python
# In instruction callable — READ ONLY
def summarize_instruction(ctx: ReadonlyContext):
    raw = ctx.state.get("market_raw", "")   # ✅ Read
    ctx.state["new_key"] = "value"           # ❌ TypeError — MappingProxyType

# In callback — READ + WRITE
def before_agent(callback_context: CallbackContext):
    raw = callback_context.state.get("market_raw", "")      # ✅ Read
    callback_context.state["new_key"] = "value"             # ✅ Write
```

---

## 13. The Full Execution Flow — Step by Step

This is the complete trace from user input to final decision:

```
USER: "Tell me about the EV market in India"
│
│  [ADK receives message, creates Session + InvocationContext]
│
▼
STEP 1: router_agent (LlmAgent)
│  • Receives: user message
│  • LLM reads instruction → decides this is a market query
│  • LLM calls: transfer_to_agent("market_research_pipeline")
│  • ADK routes to market_research_pipeline
│
▼
STEP 2: market_research_pipeline (SequentialAgent)
│  • Not an LLM — just orchestrates sub-agents in order
│  • Calls sub_agents[0].run_async(ctx) → market_agent
│
▼
STEP 3: market_agent (LlmAgent)
│  • model: gpt-3.5-turbo
│  • instruction: "You are a market research analyst..."
│  • tools: [web_search]
│  • LLM decides to call web_search("EV market outlook India 2025")
│  • log_event("web_search_request", {"query": ...})  ← logged to token_log.jsonl
│  • Tavily API returns 5 search results
│  • log_event("web_search_response", {"results_count": 5})
│  • LLM synthesizes: "India's EV market is growing at 49% CAGR..."
│  • ADK calls __maybe_save_output_to_state()
│  • state["market_raw"] = "India's EV market..." (via state_delta)
│
▼
STEP 4: market_summarizer (LlmAgent)
│  • instruction: summarize_instruction(ctx)  ← CALLABLE
│  • ADK calls summarize_instruction(ReadonlyContext)
│  • Function reads: ctx.state.get("market_raw")
│  • count_tokens(raw) → 525 tokens
│  • log_event("token_input", {"stage": "market", "raw_tokens": 525})
│  • Returns: "Compress this:\n[full 525-token text]"
│  • LLM compresses to ≤180 tokens
│  • state["market_summary"] = "India EV market: 49% CAGR, key players..."
│
▼
STEP 5: finance_agent (LlmAgent)
│  • Calls web_search if needed
│  • state["finance_raw"] = "EV charging station cost: ₹15-30L..."
│
▼
STEP 6: finance_summarizer
│  • Reads state["finance_raw"]
│  • count_tokens(raw) → 448 tokens
│  • log_event("token_input", {"stage": "finance", "raw_tokens": 448})
│  • state["finance_summary"] = compressed financial data
│
▼
STEP 7-10: policy_agent → policy_summarizer → tech_agent → tech_summarizer
│  (Same pattern for each pair)
│
▼
STEP 11: risk_agent → risk_summarizer
│  • state["risk_summary"] = compressed risk data
│
▼
STEP 12: decision_agent (LlmAgent)
│  • Reads all *_summary keys from state (via instruction or implicit context)
│  • No tools needed — synthesizes from state
│  • LLM output: "Verdict: YES\nTop 3 Reasons:..."
│  • state["decision"] = final verdict
│
▼
FINAL RESPONSE returned to user via router_agent
```

---

## 14. State Scopes — session vs user vs app vs temp

**ADK Source:** `sessions/state.py`, `sessions/_session_util.py`

ADK's state system has four scopes, identified by key prefix:

```python
class State:
    APP_PREFIX  = "app:"    # Shared across ALL users of this application
    USER_PREFIX = "user:"   # Shared across all sessions for ONE user
    TEMP_PREFIX = "temp:"   # Discarded after the current invocation ends
    # (no prefix)           # Session scope — lives for this conversation only
```

### Scope Comparison

| Prefix | Scope | Example Use |
|--------|-------|-------------|
| `"app:"` | Global — all users | App-wide configuration, shared reference data |
| `"user:"` | Per-user — all sessions | User preferences, historical summaries |
| (none) | Per-session | Current conversation data (what this project uses) |
| `"temp:"` | Per-invocation | Intermediate data, NOT persisted to database |

### How This Project Uses State

All keys in this project use **no prefix** (session scope):
```python
output_key="market_raw"      # session scope — only in this conversation
output_key="market_summary"  # session scope — shared within this pipeline run
```

**Temp keys example (not used in this project but important to know):**
```python
# In a callback:
ctx.state["temp:intermediate_calc"] = some_value
# This value is available during this invocation but will NOT
# be persisted to the database by the session service.
```

### How State Persistence Works

```
InMemorySessionService    → State lives in RAM, lost on restart (development only)
SQLiteSessionService      → State persisted to .db file (lightweight production)
DatabaseSessionService    → State persisted to PostgreSQL/MySQL (full production)
VertexAiSessionService    → State persisted to Google Cloud (GCP production)
```

This project currently runs with `InMemorySessionService` (ADK's default when using `adk web`). For production, swap to `SQLiteSessionService` or `DatabaseSessionService`.

---

## 15. The Dynamic Instruction Pattern — This Project's Key Technique

This section consolidates the most important pattern in this codebase — the summarizer design — and explains every decision.

### The Pattern

```python
# market_summarizer.py — Complete annotated explanation

from google.adk.agents import Agent
from google.adk.models import LiteLlm
from ..logging.token_meter import count_tokens
from ..logging.logger import log_event

MODEL = LiteLlm(model="gpt-3.5-turbo")

def summarize_instruction(ctx):
    # ① ctx is ReadonlyContext — provided by ADK at runtime
    # ② ctx.state is the session state dict (read-only MappingProxyType here)
    raw = ctx.state.get("market_raw", "")
    #              ↑ This key was written by market_agent via output_key="market_raw"

    # ③ Count tokens BEFORE compression — for observability
    raw_tokens = count_tokens(raw)

    # ④ Build the full instruction with content embedded
    instruction = f"""
You compress research output.

Rules:
- Keep only facts, numbers, trends
- Remove fluff and duplication
- <= 180 tokens

TEXT:
{raw}
"""
    # ↑ The actual content to compress is injected into the system prompt.
    # This is a deliberate architectural choice. The alternative would be
    # to rely on ADK's message history, but that's less predictable for
    # a compression task — we want the summarizer to see exactly this text.

    # ⑤ Log before the LLM call — after this function returns, the LLM runs
    log_event("token_input", {
        "stage": "market",
        "raw_tokens": raw_tokens,
    })

    return instruction  # ← ADK uses this as the system prompt

market_summarizer = Agent(
    name="market_summarizer",
    model=MODEL,
    description="Compresses market research output.",
    instruction=summarize_instruction,  # ← Function reference, NOT function call
    output_key="market_summary",        # ← LLM output → state["market_summary"]
)
```

### Why Not Use ADK's Built-in Placeholder Injection?

ADK supports `{variable_name}` placeholders in string instructions:
```python
instruction="Compress this: {market_raw}"
# ADK would substitute ctx.state["market_raw"] at runtime
```

This works, but the callable approach is preferred here because:
1. You can run Python logic (token counting, logging) before the LLM call
2. You can access the value once and reuse it for multiple purposes
3. You have full control over formatting and error handling
4. You can set defaults (`ctx.state.get("market_raw", "")` vs raising KeyError)

### The Token Efficiency Result

```
Without this pattern (naïve approach):
  All 5 research outputs in context for decision_agent
  = ~2,500 tokens of raw text
  × 11 LLM calls in pipeline
  = Context grows with every step

With this pattern:
  Each raw output immediately compressed after generation
  = ~900 tokens of summaries for decision_agent
  = ~65% token reduction
  = Lower cost + faster responses + more reliable (less context overflow)
```

---

## 16. Complete Import Reference

Every ADK import used in this project, with explanations:

```python
# ── AGENTS ──────────────────────────────────────────────────────────────────
from google.adk.agents import Agent
# LlmAgent alias. The primary agent class. Use for any agent that calls an LLM.

from google.adk.agents import SequentialAgent
# Workflow agent. Runs sub_agents in order. No LLM of its own.

# NOT USED IN ACTIVE CODE but available:
# from google.adk.agents import ParallelAgent   — runs sub_agents simultaneously
# from google.adk.agents import LoopAgent       — runs sub_agents in a loop
# from google.adk.agents import BaseAgent       — base class for custom agents
# from google.adk.agents import InvocationContext  — runtime context object


# ── MODELS ──────────────────────────────────────────────────────────────────
from google.adk.models import LiteLlm
# Wrapper for non-Gemini models (OpenAI, Anthropic, etc.) via LiteLLM library.
# Usage: LiteLlm(model="gpt-3.5-turbo")


# ── TOOLS ───────────────────────────────────────────────────────────────────
from google.adk.tools.function_tool import FunctionTool
# Wraps a Python function into an ADK-compatible tool.
# Usage: FunctionTool(my_function)
# ADK reads function signature + docstring to generate the LLM tool schema.


# ── CONTEXT OBJECTS (what you receive in callbacks/instructions) ─────────────
# ReadonlyContext  — what you get in callable instructions
#   ctx.state         → MappingProxyType (read-only)
#   ctx.user_content  → The user's input
#   ctx.session       → Session object
#   ctx.user_id       → User ID string

# CallbackContext  — what you get in agent/model/tool callbacks
#   ctx.state         → State object (read + write)
#   ctx.save_artifact()  → Save a file to artifact store
#   ctx.load_artifact()  → Load a file from artifact store
```

### Third-Party Libraries Used for ADK Support

```python
import tiktoken
# OpenAI's tokenizer library.
# Used in token_meter.py to count tokens BEFORE sending to LLM.
# cl100k_base encoding = same tokenizer as GPT-3.5-turbo and GPT-4.
# Lets you enforce token limits without making an API call.

import requests
# HTTP client. Used in web_search.py to call the Tavily API.
# Standard Python library — not ADK-specific.

import os
# Used to read TAVILY_API_KEY and OPENAI_API_KEY from environment variables.
# ADK does NOT manage API keys — you must set env vars yourself.
```

---

## 17. Common Pitfalls & ADK Gotchas

These are the most common mistakes when learning ADK, annotated with examples from this codebase.

---

### ❌ Pitfall 1: `root_agent` Must Be Exactly That Name

```python
# ✅ Correct — ADK finds this
root_agent = router_agent

# ❌ Wrong — ADK won't find these
my_agent = router_agent
main_agent = router_agent
```

---

### ❌ Pitfall 2: Calling the Instruction Function Instead of Passing It

```python
# ✅ Correct — pass the function reference
market_summarizer = Agent(
    instruction=summarize_instruction,   # No parentheses!
)

# ❌ Wrong — this calls the function at definition time, not at runtime
market_summarizer = Agent(
    instruction=summarize_instruction(), # Parentheses = called immediately
    # ctx doesn't exist yet at import time → crash
)
```

---

### ❌ Pitfall 3: `decision_agent` Instruction References Wrong State Keys

```python
# Current instruction in decision_agent.py references:
# "- market_raw"
# "- finance_raw"
# But by the time decision_agent runs, these have been replaced by summaries.

# ✅ Fix: Update the instruction to reference the correct keys
instruction="""
You are the final decision maker.

Inputs available in state:
- market_summary      ← correct
- finance_summary     ← correct
- policy_summary      ← correct
- tech_summary        ← correct
- risk_summary        ← correct

Task: Decide business viability...
"""
```

Note: Even though the instruction says those keys, decision_agent's LLM may still read the actual state content via ADK's state injection — but making the instruction accurate is best practice.

---

### ❌ Pitfall 4: Agent Names Must Be Unique in the Tree

```python
# ❌ Wrong — two agents named the same thing
agent1 = Agent(name="researcher", ...)
agent2 = Agent(name="researcher", ...)  # ADK will warn/error

# ✅ Correct — unique names
agent1 = Agent(name="market_researcher", ...)
agent2 = Agent(name="finance_researcher", ...)
```

---

### ❌ Pitfall 5: `state_delta` vs `session.state` — Reading Right After Writing

```python
# In the SAME agent turn, if you write to state in a callback
# and then read it immediately in the same callback,
# it works because State reads delta first.

# But if you try to read from session.state (not State object):
session.state["my_key"]   # May not see delta yet
ctx.state["my_key"]       # Correctly sees delta (uses State object)

# Always use ctx.state, not ctx.session.state directly.
```

---

### ❌ Pitfall 6: `temp:` Keys Are Not Persisted

```python
# These keys disappear after the invocation ends
ctx.state["temp:my_data"] = value   # Only lives during this invocation

# If you need data in the next turn, use session scope:
ctx.state["my_data"] = value        # Persists in session
```

---

### ❌ Pitfall 7: `log_event` Receives a `mappingproxy` — Always Copy

```python
# ❌ Wrong — MappingProxyType cannot be JSON serialized
log_event("event", ctx.state)

# ✅ Correct (as done in logger.py) — copy to regular dict first
def log_event(event: str, payload: dict):
    payload = dict(payload)   # ← Converts mappingproxy to plain dict
    payload["event"] = event
    ...
```

This is why `logger.py` does `payload = dict(payload)` — it guards against `MappingProxyType` objects being passed from callbacks.

---

### ❌ Pitfall 8: `callbacks.py` `before_model` Is Not Wired

```python
# callbacks.py defines:
def before_model(ctx):
    ctx.messages = trim_context(ctx.messages)

# But no agent has:
# before_model_callback=before_model

# To activate it:
market_agent = Agent(
    ...
    before_model_callback=before_model,
)
```

---

### ✅ ADK Agent Development Checklist

Use this when building new ADK agents:

- [ ] `root_agent` variable exists in `agent.py`
- [ ] All agent `name` values are unique and are valid Python identifiers
- [ ] `description` is precise and specific (used for routing decisions)
- [ ] If using dynamic instruction, pass function **reference** (no parentheses)
- [ ] `output_key` keys match what downstream agents read from `ctx.state`
- [ ] `OPENAI_API_KEY` set in `.env` if using `LiteLlm`
- [ ] `TAVILY_API_KEY` set in `.env` if using web_search
- [ ] Callbacks are wired to agents via `before_model_callback=`, etc.
- [ ] Agents needing production persistence use `SQLiteSessionService` or `DatabaseSessionService`

---

*This document covers every ADK concept used in Smart Market Advisor V2 and is intended as both a code walkthrough and a reusable learning reference for Google ADK development.*