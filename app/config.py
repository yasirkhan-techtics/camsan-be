import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()
poppler_path = os.getenv("POPPLER_PATH")

class Settings(BaseSettings):
    """Central application configuration."""

    database_url: str = Field(..., alias="DATABASE_URL")
    uploads_dir: str = Field("uploads", alias="UPLOADS_DIR")
    poppler_path: Optional[str] = Field(poppler_path, alias="POPPLER_PATH")
    gemini_api_key: Optional[str] = Field(None, alias="GEMINI_API_KEY")
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    template_match_threshold: float = 0.7
    label_scale_range: Tuple[float, float] = (0.6, 1.3)
    max_upload_size: int = 50 * 1024 * 1024  # 50 MB
    frontend_base_url: Optional[AnyHttpUrl] = None

    class Config:
        # Try to find .env in current directory or app directory
        current_dir = Path.cwd()
        if (current_dir / ".env").exists():
            env_file = ".env"
        elif (current_dir / "app" / ".env").exists():
            env_file = "app/.env"
        elif (current_dir.parent / "app" / ".env").exists():
            env_file = str(current_dir.parent / "app" / ".env")
        else:
            env_file = ".env"  # Default fallback
        
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    settings = Settings()
    print(f"⚙️ Settings loaded - POPPLER_PATH: {settings.poppler_path}")
    return settings


