from google.adk.agents import Agent
from google.adk.models import LiteLlm

from ..logging.token_meter import count_tokens
from ..logging.logger import log_event

MODEL = LiteLlm(model="gpt-3.5-turbo")


def summarize_instruction(ctx):
    raw = ctx.state.get("finance_raw", "")
    raw_tokens = count_tokens(raw)

    instruction = f"""
You compress financial analysis.

Rules:
- Keep only costs, margins, ROI, risks
- Remove fluff
- <= 180 tokens

TEXT:
{raw}
"""

    log_event("token_input", {
        "stage": "finance",
        "raw_tokens": raw_tokens,
    })

    return instruction


finance_summarizer = Agent(
    name="finance_summarizer",
    model=MODEL,
    description="Compresses finance research output.",
    instruction=summarize_instruction,
    output_key="finance_summary",
)




# from google.adk.agents import Agent
# from google.adk.models import LiteLlm
# from ..logging.token_meter import count_tokens
# from ..logging.logger import log_event

# MODEL = LiteLlm(model="gpt-3.5-turbo")

# def summarize_instruction(ctx):
#     raw = ctx.state.get("finance_raw", "")
#     return f"""
# Summarize the following finance analysis.

# Rules:
# - Keep only costs, ROI, margins
# - Remove fluff
# - <= 150 tokens

# TEXT:
# {raw}
# """

# def summarize_and_gate(ctx, response: str):
#     raw = ctx.state.get("finance_raw", "")
#     raw_tokens = count_tokens(raw)
#     summary_tokens = count_tokens(response)

#     log_event("finance_compression", {
#         "raw_tokens": raw_tokens,
#         "summary_tokens": summary_tokens,
#         "reduction_%": round((1 - summary_tokens / max(raw_tokens,1)) * 100, 2)
#     })

#     ctx.state["finance_summary"] = response
#     ctx.state.pop("finance_raw", None)

# finance_summarizer = Agent(
#     name="finance_summarizer",
#     model=MODEL,
#     instruction=summarize_instruction,
#     output_key="finance_summary",
#     after_response_callback=summarize_and_gate,
# )

