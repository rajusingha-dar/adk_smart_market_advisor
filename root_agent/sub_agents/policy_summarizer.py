from google.adk.agents import Agent
from google.adk.models import LiteLlm

from ..logging.token_meter import count_tokens
from ..logging.logger import log_event

MODEL = LiteLlm(model="gpt-3.5-turbo")

def summarize_instruction(ctx):
    raw = ctx.state.get("policy_raw", "")
    raw_tokens = count_tokens(raw)

    instruction = f"""
You compress policy research.

Rules:
- Keep only incentives, regulations, risks
- Remove fluff and repetition
- <= 180 tokens

TEXT:
{raw}
"""

    log_event("token_input", {
        "stage": "policy",
        "raw_tokens": raw_tokens,
    })

    return instruction


policy_summarizer = Agent(
    name="policy_summarizer",
    model=MODEL,
    description="Compresses policy research output.",
    instruction=summarize_instruction,
    output_key="policy_summary",
)
