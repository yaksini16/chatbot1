"""
Unified Configuration for PRAMAN v4 / Legal Metrology Unified Backend
Bridging core.config and CB2 settings.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from core import config as core_cfg



class Settings(BaseSettings):
    APP_NAME: str = "PRAMAN v4 Legal Metrology Backend"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # OCR Settings
    OCR_PRIMARY_ENGINE: str = "paddleocr"
    OCR_SECONDARY_ENGINE: str = "easyocr"
    OCR_THIRD_ENGINE: str = "surya"
    OCR_CONFIDENCE_THRESHOLD: float = core_cfg.OCR_CONFIDENCE_THRESHOLD
    EXTRACTION_CONFIDENCE_THRESHOLD: float = 0.60
    VLM_MODEL_ID: str = core_cfg.VLM_MODEL_ID

    # Vision quality
    MIN_SHARPNESS_LAPLACIAN: float = float(core_cfg.MIN_SHARPNESS_LAPLACIAN)
    MAX_GLARE_RATIO: float = float(core_cfg.GLARE_THRESHOLD)
    AUTHENTICITY_THRESHOLD: float = 0.85

    # Storage & Database
    DATABASE_URL: str = core_cfg.DATABASE_URL
    UPLOAD_DIR: str = core_cfg.UPLOAD_DIR

    # Security
    JWT_SECRET_KEY: str = core_cfg.SECRET_KEY
    JWT_ALGORITHM: str = core_cfg.ALGORITHM
    JWT_EXPIRY_MINUTES: int = core_cfg.ACCESS_TOKEN_EXPIRE_MINUTES

    # Gemini LLM Settings
    # Reads GEMINI_API_KEY from environment or .env without logging or printing
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-3.6-flash"


    @property
    def is_gemini_configured(self) -> bool:
        """
        Detects whether GEMINI_API_KEY is configured.
        SECURITY: Returns boolean only; never logs or prints the API key value.
        """
        key = self.GEMINI_API_KEY if self.GEMINI_API_KEY is not None else os.environ.get("GEMINI_API_KEY")
        return bool(key and str(key).strip())



    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

