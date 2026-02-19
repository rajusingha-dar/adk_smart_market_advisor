from google.adk.agents import Agent
from google.adk.models import LiteLlm

MODEL = LiteLlm(model="gpt-3.5-turbo")

decision_agent = Agent(
    name="decision_agent",
    model=MODEL,
    description="Synthesizes research and decides business viability.",
    instruction="""
You are the final decision maker.

Inputs available in state:
- market_raw
- finance_raw

Task:
1. Decide if the business/topic is viable.
2. Give:
   - Verdict (Yes/No/Maybe)
   - Top 3 reasons
   - Clear next steps

Write a short, structured decision.
""",
    output_key="decision",
)
