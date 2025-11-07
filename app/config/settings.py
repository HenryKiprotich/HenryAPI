from pydantic_settings import BaseSettings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    DEBUG: bool = True
    DATABASE_URL: str  # For asyncpg (raw SQL)
    SQLALCHEMY_DATABASE_URL: str  # For SQLAlchemy ORM
    huggingfacehub_api_token: str
    google_api_key: str
    deepseek_api_key: str
    deepseek_api_base: str = "https://api.deepseek.com"
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"

    class Config:
        env_file = ".env"

try:
    settings = Settings()
    logger.info("Settings loaded successfully.")
except Exception as e:
    logger.exception(f"Error loading settings: {e}")
    raise