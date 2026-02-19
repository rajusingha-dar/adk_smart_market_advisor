from google.adk.agents import Agent
from google.adk.models import LiteLlm

from ..logging.token_meter import count_tokens
from ..logging.logger import log_event

MODEL = LiteLlm(model="gpt-3.5-turbo")


def summarize_instruction(ctx):
    raw = ctx.state.get("market_raw", "")
    raw_tokens = count_tokens(raw)

    instruction = f"""
You compress research output.

Rules:
- Keep only facts, numbers, trends
- Remove fluff and duplication
- <= 180 tokens

TEXT:
{raw}
"""

    # LOG here (do not store)
    log_event("token_input", {
        "stage": "market",
        "raw_tokens": raw_tokens,
    })

    return instruction


market_summarizer = Agent(
    name="market_summarizer",
    model=MODEL,
    description="Compresses market research output.",
    instruction=summarize_instruction,
    output_key="market_summary",
)



# from google.adk.agents import Agent
# from google.adk.models import LiteLlm
# from ..logging.token_meter import count_tokens
# from ..logging.logger import log_event

# MODEL = LiteLlm(model="gpt-3.5-turbo")

# def summarize_instruction(ctx):
#     raw = ctx.state.get("market_raw", "")
#     raw_tokens = count_tokens(raw)

#     return f"""
# Summarize the following market research.

# Rules:
# - Keep only facts, numbers, trends
# - Remove fluff
# - <= 180 tokens

# TEXT:
# {raw}
# """

# def summarize_and_gate(ctx, response: str):
#     raw = ctx.state.get("market_raw", "")
#     raw_tokens = count_tokens(raw)
#     summary_tokens = count_tokens(response)

#     # LOG
#     log_event("market_compression", {
#         "raw_tokens": raw_tokens,
#         "summary_tokens": summary_tokens,
#         "reduction_%": round((1 - summary_tokens / max(raw_tokens,1)) * 100, 2)
#     })

#     # 🔥 GATE
#     ctx.state["market_summary"] = response
#     ctx.state.pop("market_raw", None)   # DELETE RAW

# market_summarizer = Agent(
#     name="market_summarizer",
#     model=MODEL,
#     instruction=summarize_instruction,
#     output_key="market_summary",
#     after_response_callback=summarize_and_gate,
# )
