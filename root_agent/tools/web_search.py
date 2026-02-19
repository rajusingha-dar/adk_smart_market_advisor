import os
import requests
from google.adk.tools.function_tool import FunctionTool
from ..logging.logger import log_event

def _web_search(query: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": 5,
        "search_depth": "advanced",
    }

    log_event("web_search_request", {"query": query})

    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()

    results = r.json().get("results", [])
    text = "\n".join([item.get("content", "") for item in results])

    log_event("web_search_response", {
        "query": query,
        "results_count": len(results),
    })

    return text or "No results found."

web_search = FunctionTool(_web_search)
