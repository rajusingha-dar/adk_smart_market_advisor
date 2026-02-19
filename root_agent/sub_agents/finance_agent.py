from google.adk.agents import Agent
from google.adk.models import LiteLlm
from ..tools.web_search import web_search

MODEL = LiteLlm(model="gpt-3.5-turbo")

finance_agent = Agent(
    name="finance_agent",
    model=MODEL,
    description=(
        "Financial research specialist. Analyzes costs, revenue models, "
        "margins, pricing, and ROI for any business or industry."
    ),
    instruction="""
You are a financial analyst.

Your job:
- Identify cost structure, margins, revenue models, and ROI for the topic.
- If numbers are not available, use web_search.
- Explain assumptions clearly.
- Output practical, business-oriented insights.
""",
    tools=[web_search],
    output_key="finance_raw",
)
