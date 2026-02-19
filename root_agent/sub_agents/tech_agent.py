from google.adk.agents import Agent
from google.adk.models import LiteLlm
from ..tools.web_search import web_search

MODEL = LiteLlm(model="gpt-3.5-turbo")

tech_agent = Agent(
    name="tech_agent",
    model=MODEL,
    description="Researches EV technology and infrastructure.",
    instruction="""
You are a technology market analyst.

Task:
Research technology and infrastructure trends for the topic.

Focus on:
- Charging infrastructure
- Battery technology
- Manufacturing processes
- Innovation and cost reductions
""",
    tools=[web_search],
    output_key="tech_raw",
)
