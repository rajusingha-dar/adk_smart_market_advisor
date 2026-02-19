from google.adk.agents import Agent
from google.adk.models import LiteLlm

from ..logging.token_meter import count_tokens
from ..logging.logger import log_event

MODEL = LiteLlm(model="gpt-3.5-turbo")

def summarize_instruction(ctx):
    raw = ctx.state.get("tech_raw", "")
    raw_tokens = count_tokens(raw)

    instruction = f"""
You compress technology research.

Rules:
- Keep only tech trends, infra, innovations
- Remove fluff
- <= 180 tokens

TEXT:
{raw}
"""

    log_event("token_input", {
        "stage": "tech",
        "raw_tokens": raw_tokens,
    })

    return instruction


tech_summarizer = Agent(
    name="tech_summarizer",
    model=MODEL,
    description="Compresses technology research output.",
    instruction=summarize_instruction,
    output_key="tech_summary",
)
