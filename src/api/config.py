"""
API configuration settings.
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Info
    app_name: str = "FraudGuard API"
    app_version: str = "1.0.0"
    app_description: str = "Real-time fraud detection for mobile money transactions"
    
    # Model paths
    model_path: str = "models/fraud_model.pkl"
    feature_names_path: str = "models/feature_names.json"
    
    # Model config
    default_threshold: float = 0.75
    
    # API config
    api_prefix: str = "/api/v1"
    debug: bool = False
    
    # Rate limiting (future)
    rate_limit_enabled: bool = False
    rate_limit_per_minute: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()