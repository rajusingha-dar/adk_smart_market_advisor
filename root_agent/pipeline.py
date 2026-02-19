# from google.adk.agents import SequentialAgent

# from .sub_agents.market_agent import market_agent
# from .sub_agents.finance_agent import finance_agent
# from .sub_agents.policy_agent import policy_agent
# from .sub_agents.tech_agent import tech_agent
# from .sub_agents.risk_agent import risk_agent
# from .sub_agents.summarizer_agent import summarizer_agent
# from .sub_agents.decision_agent import decision_agent

# market_research_pipeline = SequentialAgent(
#     name="smart_pipeline",
#     description="Sequential EV business analysis pipeline",
#     agents=[
#         market_agent,
#         finance_agent,
       
#     ],
# )


#  policy_agent,
#         tech_agent,
#         risk_agent,
#         summarizer_agent,
#         decision_agent,


# from google.adk.agents import SequentialAgent
# from .sub_agents.market_agent import market_agent
# from .sub_agents.finance_agent import finance_agent
# from .sub_agents.decision_agent import decision_agent
# from .memory.summarizer_gate import make_summarizer

# from .memory.context_pruner import prune_context

# market_summary = make_summarizer("market_summarizer")
# finance_summary = make_summarizer("finance_summarizer")

# market_research_pipeline = SequentialAgent(
#     name="market_research_pipeline",
#     description="Market → Compress → Finance → Compress → Decision",
#     sub_agents=[
#         market_agent,
#         market_summary,
#         finance_agent,
#         finance_summary,
#         decision_agent,
#     ],
#     after_sub_agent_callback=prune_context
# )


from google.adk.agents import SequentialAgent
from .sub_agents.market_agent import market_agent
from .sub_agents.finance_agent import finance_agent
from .sub_agents.decision_agent import decision_agent
from .sub_agents.market_summarizer import market_summarizer
from .sub_agents.finance_summarizer import finance_summarizer
from .sub_agents.policy_agent import policy_agent
from .sub_agents.policy_summarizer import policy_summarizer 
from .sub_agents.tech_agent import tech_agent
from .sub_agents.tech_summarizer import tech_summarizer
from .sub_agents.risk_agent import risk_agent
from .sub_agents.risk_summarizer import risk_summarizer

# market_research_pipeline = SequentialAgent(
#     name="market_research_pipeline",
#     description="Market → Compress → Finance → Compress → Decision",
#     sub_agents=[
#         market_agent,
#         market_summarizer,
#         finance_agent,
#         finance_summarizer,
#         decision_agent,
#     ],
# )



market_research_pipeline = SequentialAgent(
    name="market_research_pipeline",
    description="Market → Compress → Finance → Compress → Policy → Compress → Tech → Compress → Risk → Compress → Decision",
    sub_agents=[
        market_agent,
        market_summarizer,
        finance_agent,
        finance_summarizer,
        policy_agent,
        policy_summarizer,
        tech_agent,
        tech_summarizer,
        risk_agent,
        risk_summarizer,
        decision_agent,
    ],
)
