from google.adk.agents import Agent
from google.adk.models import LiteLlm
from ..tools.web_search import web_search

MODEL = LiteLlm(model="gpt-3.5-turbo")

risk_agent = Agent(
    name="risk_agent",
    model=MODEL,
    description="Identifies business, market, and operational risks.",
    instruction="""
You are a risk analyst.

Task:
Identify major risks for the given business topic.

Focus on:
- Market risks
- Financial risks
- Regulatory risks
- Technology & execution risks
""",
    tools=[web_search],
    output_key="risk_raw",
)
