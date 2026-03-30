from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_tokens: int = 200
    device: str = "cpu"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()