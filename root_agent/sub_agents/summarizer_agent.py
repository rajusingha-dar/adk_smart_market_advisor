from google.adk.agents import Agent
from google.adk.models import LiteLlm

MODEL = LiteLlm(model="gpt-3.5-turbo")

summarizer_agent = Agent(
    name="summarizer_agent",
    model=MODEL,
    description="Compresses long tool outputs.",
    instruction="""
You are a compression agent.

Take the previous agent output and:
- Remove fluff
- Keep only key numbers, trends, and facts
- Max 120 tokens

Return a concise summary.
""",
    output_key="compressed",
)
