"""
Dependency injection for FastAPI endpoints.
"""
from functools import lru_cache
from fastapi import HTTPException, status
import time
from datetime import datetime

from src.models.predictor import FraudPredictor
from src.api.config import settings


# Global predictor instance (loaded once at startup)
_predictor_instance = None
_startup_time = None


def get_predictor() -> FraudPredictor:
    """
    Get fraud predictor instance (singleton).
    
    Returns:
        FraudPredictor instance
        
    Raises:
        HTTPException: If predictor fails to load
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        try:
            _predictor_instance = FraudPredictor(
                model_path=settings.model_path,
                feature_names_path=settings.feature_names_path,
                threshold=settings.default_threshold
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load fraud detection model: {str(e)}"
            )
    
    return _predictor_instance


def get_startup_time() -> datetime:
    """Get service startup time"""
    global _startup_time
    if _startup_time is None:
        _startup_time = datetime.now()
    return _startup_time


def get_uptime() -> float:
    """Get service uptime in seconds"""
    start = get_startup_time()
    return (datetime.now() - start).total_seconds()