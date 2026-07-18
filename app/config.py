import os
from dotenv import load_dotenv


load_dotenv()

class Settings:
    ANTHROPIC_API_KEY:str = os.getenv("ANTHROPIC_API_KEY")
    LLM_MODEL:str = os.getenv("LLM_MODEL")
    OPENAI_API_KEY:str=os.getenv("OPEN_AI_API_KEY")
    QDRANT_ENDPOINT: str = os.getenv("QDRANT_ENDPOINT", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")



settings=Settings()