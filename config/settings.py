import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    MAX_USER_TURNS = int(os.getenv("MAX_USER_TURNS", 3))
    MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", 2))
    SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-3.5-turbo")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
