from google.adk.agents import Agent
from google.adk.models import LiteLlm

from ..logging.token_meter import count_tokens
from ..logging.logger import log_event

MODEL = LiteLlm(model="gpt-3.5-turbo")

def summarize_instruction(ctx):
    raw = ctx.state.get("risk_raw", "")
    raw_tokens = count_tokens(raw)

    instruction = f"""
You compress risk analysis.

Rules:
- Keep only major risks & mitigation
- Remove fluff
- <= 150 tokens

TEXT:
{raw}
"""

    log_event("token_input", {
        "stage": "risk",
        "raw_tokens": raw_tokens,
    })

    return instruction


risk_summarizer = Agent(
    name="risk_summarizer",
    model=MODEL,
    description="Compresses risk analysis output.",
    instruction=summarize_instruction,
    output_key="risk_summary",
)
