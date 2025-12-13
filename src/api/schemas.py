"""
Pydantic schemas for request/response validation.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class TransactionRequest(BaseModel):
    """Request schema for single transaction scoring"""
    
    # Transaction identifiers
    transaction_id: Optional[str] = Field(None, description="Transaction ID (optional)")
    
    # Required transaction details
    amount: float = Field(..., gt=0, description="Transaction amount (must be positive)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    
    # Time features
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    is_weekend: int = Field(..., ge=0, le=1, description="Is weekend (0 or 1)")
    is_night: int = Field(..., ge=0, le=1, description="Is night time (0 or 1)")
    is_early_morning: int = Field(0, ge=0, le=1, description="Is early morning (0 or 1)")
    is_business_hours: int = Field(0, ge=0, le=1, description="Is business hours (0 or 1)")
    hour_sin: float = Field(..., description="Hour sine encoding")
    hour_cos: float = Field(..., description="Hour cosine encoding")
    day_sin: float = Field(..., description="Day sine encoding")
    day_cos: float = Field(..., description="Day cosine encoding")
    
    # Amount features
    amount_log: float = Field(..., description="Log-transformed amount")
    amount_sqrt: float = Field(..., description="Square root of amount")
    amount_squared: float = Field(..., description="Amount squared")
    amount_deviation: float = Field(0.0, description="Deviation from user average")
    amount_zscore: float = Field(0.0, description="Z-score of amount")
    amount_percentile: float = Field(50.0, ge=0, le=100, description="Amount percentile")
    is_round_amount: int = Field(0, ge=0, le=1, description="Is round amount")
    is_month_start: int = Field(0, ge=0, le=1, description="Is month start")
    is_month_end: int = Field(0, ge=0, le=1, description="Is month end")
    
    # Velocity features
    tx_count_1h: int = Field(0, ge=0, description="Transaction count in last 1 hour")
    tx_count_6h: int = Field(0, ge=0, description="Transaction count in last 6 hours")
    tx_count_24h: int = Field(0, ge=0, description="Transaction count in last 24 hours")
    tx_count_7d: int = Field(0, ge=0, description="Transaction count in last 7 days")
    time_since_last_tx: float = Field(3600.0, ge=0, description="Time since last transaction (seconds)")
    avg_tx_interval: float = Field(3600.0, ge=0, description="Average transaction interval")
    tx_velocity_1h: int = Field(0, ge=0, description="Transaction velocity (1h)")
    tx_velocity_24h: int = Field(0, ge=0, description="Transaction velocity (24h)")
    
    # User history features
    user_tx_count: int = Field(1, ge=1, description="User's total transaction count")
    user_avg_amount: float = Field(500.0, gt=0, description="User's average amount")
    user_std_amount: float = Field(200.0, ge=0, description="User's amount std deviation")
    amount_deviation_from_user: float = Field(0.0, description="Deviation from user's average")
    amount_2x_avg: int = Field(0, ge=0, le=1, description="Amount > 2x user average")
    amount_3x_avg: int = Field(0, ge=0, le=1, description="Amount > 3x user average")
    user_fraud_rate: float = Field(0.01, ge=0, le=1, description="User's historical fraud rate")
    
    # Receiver features
    is_new_receiver: int = Field(0, ge=0, le=1, description="Is new receiver")
    receiver_tx_count: int = Field(0, ge=0, description="Previous transactions to receiver")
    
    # Device/Location features
    device_changed_flag: int = Field(0, ge=0, le=1, description="Device changed")
    device_change_count: int = Field(0, ge=0, description="Total device changes")
    location_changed_flag: int = Field(0, ge=0, le=1, description="Location changed")
    location_change_count: int = Field(0, ge=0, description="Total location changes")
    
    # Interaction features
    high_amount_at_night: int = Field(0, ge=0, le=1, description="High amount at night")
    device_changed_unusual_amount: int = Field(0, ge=0, le=1, description="Device change + unusual amount")
    new_receiver_high_amount: int = Field(0, ge=0, le=1, description="New receiver + high amount")
    velocity_spike: int = Field(0, ge=0, le=1, description="Velocity spike detected")
    amount_velocity_interaction: float = Field(0.0, description="Amount-velocity interaction")
    
    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        if v > 10_000_000:  # 10M KES
            raise ValueError('Amount exceeds maximum allowed')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN_12345",
                "amount": 5000.0,
                "hour": 14,
                "day_of_week": 2,
                "is_weekend": 0,
                "is_night": 0,
                "is_early_morning": 0,
                "is_business_hours": 1,
                "hour_sin": -0.5,
                "hour_cos": 0.87,
                "day_sin": 0.43,
                "day_cos": -0.9,
                "amount_log": 8.52,
                "amount_sqrt": 70.71,
                "amount_squared": 25000000,
                "amount_deviation": 0.5,
                "amount_zscore": 1.2,
                "amount_percentile": 65,
                "is_round_amount": 1,
                "is_month_start": 0,
                "is_month_end": 0,
                "tx_count_1h": 1,
                "tx_count_6h": 3,
                "tx_count_24h": 5,
                "tx_count_7d": 25,
                "time_since_last_tx": 3600,
                "avg_tx_interval": 4000,
                "tx_velocity_1h": 1,
                "tx_velocity_24h": 5,
                "user_tx_count": 150,
                "user_avg_amount": 520,
                "user_std_amount": 200,
                "amount_deviation_from_user": 8.62,
                "amount_2x_avg": 1,
                "amount_3x_avg": 1,
                "user_fraud_rate": 0.01,
                "is_new_receiver": 1,
                "receiver_tx_count": 0,
                "device_changed_flag": 0,
                "device_change_count": 1,
                "location_changed_flag": 0,
                "location_change_count": 2,
                "high_amount_at_night": 0,
                "device_changed_unusual_amount": 0,
                "new_receiver_high_amount": 1,
                "velocity_spike": 0,
                "amount_velocity_interaction": 5000
            }
        }


class TransactionResponse(BaseModel):
    """Response schema for transaction scoring"""
    
    transaction_id: Optional[str] = Field(None, description="Transaction ID")
    is_fraud: bool = Field(..., description="Fraud prediction (True/False)")
    fraud_score: float = Field(..., ge=0, le=1, description="Fraud probability (0-1)")
    risk_level: str = Field(..., description="Risk level (LOW/MEDIUM/HIGH)")
    threshold: float = Field(..., description="Classification threshold used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN_12345",
                "is_fraud": True,
                "fraud_score": 0.87,
                "risk_level": "HIGH",
                "threshold": 0.75,
                "processing_time_ms": 2.5
            }
        }


class TransactionResponseDetailed(TransactionResponse):
    """Detailed response with explainability"""
    
    reason_codes: List[str] = Field(default_factory=list, description="Fraud indicator codes")
    top_features: Optional[List[Dict[str, Any]]] = Field(None, description="Top contributing features")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN_12345",
                "is_fraud": True,
                "fraud_score": 0.87,
                "risk_level": "HIGH",
                "threshold": 0.75,
                "processing_time_ms": 3.2,
                "reason_codes": [
                    "amount_much_higher_than_usual",
                    "new_receiver_high_amount",
                    "amount_2x_user_average"
                ],
                "top_features": [
                    {"feature": "amount_deviation", "value": 8.62, "importance": 0.142},
                    {"feature": "is_new_receiver", "value": 1.0, "importance": 0.068}
                ]
            }
        }


class BatchTransactionRequest(BaseModel):
    """Request schema for batch scoring"""
    
    transactions: List[TransactionRequest] = Field(..., min_length=1, max_length=1000)
    include_reasons: bool = Field(False, description="Include explainability (slower)")
    
    @field_validator('transactions')
    @classmethod
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000 transactions')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "transactions": [
                    {
                        "transaction_id": "TXN_001",
                        "amount": 500,
                        "hour": 14,
                        # ... other fields ...
                    }
                ],
                "include_reasons": False
            }
        }


class BatchTransactionResponse(BaseModel):
    """Response schema for batch scoring"""
    
    total_transactions: int = Field(..., description="Total transactions processed")
    processing_time_ms: float = Field(..., description="Total processing time")
    results: List[TransactionResponse] = Field(..., description="Individual results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_transactions": 10,
                "processing_time_ms": 25.3,
                "results": [
                    {
                        "transaction_id": "TXN_001",
                        "is_fraud": False,
                        "fraud_score": 0.15,
                        "risk_level": "LOW",
                        "threshold": 0.75,
                        "processing_time_ms": 2.5
                    }
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_path: str = Field(..., description="Model file path")
    uptime_seconds: float = Field(..., description="Service uptime")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_loaded": True,
                "model_path": "models/fraud_model.pkl",
                "uptime_seconds": 1234.56
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data",
                "detail": "Amount must be positive"
            }
        }