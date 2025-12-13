"""
API route handlers.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
import time
import pandas as pd

from src.api.schemas import (
    TransactionRequest,
    TransactionResponse,
    TransactionResponseDetailed,
    BatchTransactionRequest,
    BatchTransactionResponse,
    HealthResponse,
    ErrorResponse
)
from src.api.dependencies import get_predictor, get_uptime
from src.api.config import settings
from src.models.predictor import FraudPredictor


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and model information.
    """
    try:
        predictor = get_predictor()
        model_loaded = True
    except:
        model_loaded = False
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        version=settings.app_version,
        model_loaded=model_loaded,
        model_path=settings.model_path,
        uptime_seconds=get_uptime()
    )


@router.post(
    "/score",
    response_model=TransactionResponse,
    tags=["Fraud Detection"],
    summary="Score single transaction",
    description="Predict fraud probability for a single transaction"
)
async def score_transaction(
    request: TransactionRequest,
    predictor: FraudPredictor = Depends(get_predictor)
):
    """
    Score a single transaction for fraud.
    
    - **transaction_id**: Optional transaction identifier
    - **amount**: Transaction amount (positive number)
    - **hour**: Hour of day (0-23)
    - ... (43 features total)
    
    Returns fraud prediction with score and risk level.
    """
    start_time = time.time()
    
    try:
        # Convert request to dict
        features = request.model_dump(exclude={'transaction_id'})
        
        # Predict
        result = predictor.predict(features)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Build response
        response = TransactionResponse(
            transaction_id=request.transaction_id,
            is_fraud=result['is_fraud'],
            fraud_score=result['fraud_score'],
            risk_level=result['risk_level'],
            threshold=result['threshold'],
            processing_time_ms=processing_time_ms
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/score/detailed",
    response_model=TransactionResponseDetailed,
    tags=["Fraud Detection"],
    summary="Score transaction with explainability",
    description="Predict fraud with detailed reasons and feature contributions"
)
async def score_transaction_detailed(
    request: TransactionRequest,
    predictor: FraudPredictor = Depends(get_predictor)
):
    """
    Score a transaction with detailed explainability.
    
    Returns prediction plus:
    - **reason_codes**: Human-readable fraud indicators
    - **top_features**: Most important features contributing to score
    
    Note: Slightly slower than basic /score endpoint due to explainability calculations.
    """
    start_time = time.time()
    
    try:
        # Convert request to dict
        features = request.model_dump(exclude={'transaction_id'})
        
        # Predict with reasons
        result = predictor.predict_with_reasons(features, top_n=5)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Build response
        response = TransactionResponseDetailed(
            transaction_id=request.transaction_id,
            is_fraud=result['is_fraud'],
            fraud_score=result['fraud_score'],
            risk_level=result['risk_level'],
            threshold=result['threshold'],
            processing_time_ms=processing_time_ms,
            reason_codes=result['reason_codes'],
            top_features=result.get('reasons', [])[:5]  # Top 5 features
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/score/batch",
    response_model=BatchTransactionResponse,
    tags=["Fraud Detection"],
    summary="Score multiple transactions",
    description="Batch score up to 1000 transactions at once"
)
async def score_batch(
    request: BatchTransactionRequest,
    predictor: FraudPredictor = Depends(get_predictor)
):
    """
    Score multiple transactions in a single request.
    
    - **transactions**: List of transactions (max 1000)
    - **include_reasons**: Whether to include explainability (slower)
    
    Returns results for all transactions with total processing time.
    """
    start_time = time.time()
    
    try:
        # Convert transactions to DataFrame
        transactions_data = [tx.model_dump(exclude={'transaction_id'}) for tx in request.transactions]
        df = pd.DataFrame(transactions_data)
        
        # Batch predict
        results_df = predictor.batch_predict(df, include_reasons=request.include_reasons)
        
        # Build responses
        results = []
        for i, (idx, row) in enumerate(results_df.iterrows()):
            tx_id = request.transactions[i].transaction_id
            
            result = TransactionResponse(
                transaction_id=tx_id,
                is_fraud=bool(row['is_fraud']),
                fraud_score=float(row['fraud_score']),
                risk_level=row['risk_level'],
                threshold=predictor.threshold,
                processing_time_ms=0  # Individual time not tracked in batch
            )
            results.append(result)
        
        # Calculate total processing time
        total_processing_time_ms = (time.time() - start_time) * 1000
        
        response = BatchTransactionResponse(
            total_transactions=len(request.transactions),
            processing_time_ms=total_processing_time_ms,
            results=results
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get(
    "/model/info",
    tags=["Model"],
    summary="Get model information",
    description="Returns information about the loaded fraud detection model"
)
async def get_model_info(predictor: FraudPredictor = Depends(get_predictor)):
    """
    Get information about the loaded model.
    
    Returns model metadata, feature count, threshold, etc.
    """
    try:
        info = predictor.get_model_info()
        return JSONResponse(content=info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )