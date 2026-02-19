from google.adk.agents import Agent
from google.adk.models import LiteLlm

MODEL = LiteLlm(model="gpt-3.5-turbo")

def make_summarizer(name: str):
    return Agent(
        name=name,
        model=MODEL,
        description="Compresses long tool outputs.",
        instruction="""
You compress the previous agent output.

Rules:
- Keep only key facts, numbers, and trends
- Remove fluff and duplication
- Max 120 tokens
""",
        output_key=f"{name}_summary",
    )
