from google.adk.agents import Agent
from google.adk.models import LiteLlm
from ..tools.web_search import web_search

MODEL = LiteLlm(model="gpt-3.5-turbo")

policy_agent = Agent(
    name="policy_agent",
    model=MODEL,
    description="Researches EV government policies, incentives, and regulations.",
    instruction="""
You are a policy research analyst.

Task:
Research government, state, and regulatory policies for the given topic.

Focus on:
- Subsidies and incentives
- Regulations and compliance
- National & state programs
- Risks from policy changes

Write in clear bullet points.
""",
    tools=[web_search],
    output_key="policy_raw",
)
