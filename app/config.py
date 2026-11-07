import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    recursion_limit: int = int(os.getenv("RECURSION_LIMIT", "14"))


settings = Settings()
