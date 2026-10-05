from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "agent-playground"
    env: str = Field(default="dev", pattern="^(dev|staging|prod)$")
    log_level: str = "INFO"
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    request_timeout: float = Field(default=30.0, gt=0, le=300)
