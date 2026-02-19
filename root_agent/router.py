from google.adk.agents import Agent
from google.adk.models import LiteLlm
from .pipeline import market_research_pipeline


MODEL = LiteLlm(model="gpt-3.5-turbo")


router_agent = Agent(
    name="router_agent",
    model=MODEL,
    description=(
        "Entry point for the system. Understands user intent and decides whether "
        "to route to the market research pipeline or respond directly."
    ),
    instruction="""
You are the main router.

Behavior rules:

1. If the user greets (hi, hello, hey, good morning, etc):
   - Greet them politely.
   - Say you can help with market, business, or industry research on any topic.
   - Ask what topic they want to explore.

2. If the user asks about:
   - markets
   - industries
   - business feasibility
   - trends
   - costs, margins, ROI
   - competitors
   - growth, opportunities, risks

   Then:
   - Route the request to the market_research_pipeline.

3. If the query is unclear:
   - Ask a clarifying question.
""",
    sub_agents=[market_research_pipeline],
)
