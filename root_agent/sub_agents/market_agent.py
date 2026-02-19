from google.adk.agents import Agent
from google.adk.models import LiteLlm
from ..tools.web_search import web_search

MODEL = LiteLlm(model="gpt-3.5-turbo")

market_agent = Agent(
    name="market_agent",
    model=MODEL,
    description=
        "Market research specialist. Gathers market size, growth, trends, "
        "key players, and demand signals for any industry or topic."
    ,
    instruction="""
You are a market research analyst.

Your job:
- Understand the user's topic.
- Research market size, growth rate, trends, major players, and demand drivers.
- If data is missing, use web_search to fetch it.
- Return clear, structured findings.
""",
    tools=[web_search],
    output_key="market_raw",
)
