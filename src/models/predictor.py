"""
Fraud prediction class for production use.

Loads trained model and makes predictions on new transactions.
"""

import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union


class FraudPredictor:
    """Make fraud predictions on new transactions"""
    
    def __init__(
        self,
        model_path: str = "models/fraud_model.pkl",
        feature_names_path: str = "models/feature_names.json",
        threshold: float = 0.5
    ):
        """
        Args:
            model_path: Path to trained model pickle file
            feature_names_path: Path to feature names JSON
            threshold: Classification threshold (0-1)
        """
        self.model_path = Path(model_path)
        self.feature_names_path = Path(feature_names_path)
        self.threshold = threshold
        
        # Load model and feature names
        self.model = self._load_model()
        self.feature_names = self._load_feature_names()
    
    def _load_model(self):
        """Load trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    def _load_feature_names(self) -> List[str]:
        """Load feature names"""
        if not self.feature_names_path.exists():
            raise FileNotFoundError(f"Feature names not found at {self.feature_names_path}")
        
        with open(self.feature_names_path, 'r') as f:
            feature_names = json.load(f)
        
        return feature_names
    
    def predict_proba(self, features: Union[pd.DataFrame, Dict[str, Any]]) -> float:
        """
        Get fraud probability for transaction.
        
        Args:
            features: Dictionary or DataFrame with feature values
        
        Returns:
            Fraud probability (0-1)
        """
        # Convert to DataFrame if dict
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        # Ensure all features present
        missing_features = set(self.feature_names) - set(features.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select and order features
        X = features[self.feature_names]
        
        # Predict
        proba = self.model.predict_proba(X)[:, 1]
        
        return float(proba[0]) if len(proba) == 1 else proba.tolist()
    
    def predict(
        self, 
        features: Union[pd.DataFrame, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Predict fraud with full details.
        
        Args:
            features: Feature values
        
        Returns:
            Dictionary with:
                - is_fraud: Boolean prediction
                - fraud_score: Probability (0-1)
                - risk_level: "low", "medium", "high"
                - threshold: Used threshold
        """
        # Get probability
        fraud_score = self.predict_proba(features)
        
        # Classify
        is_fraud = fraud_score >= self.threshold
        
        # Risk level
        if fraud_score < 0.3:
            risk_level = "low"
        elif fraud_score < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return {
            'is_fraud': bool(is_fraud),
            'fraud_score': float(fraud_score),
            'risk_level': risk_level,
            'threshold': self.threshold
        }
    
    def predict_with_reasons(
        self,
        features: Union[pd.DataFrame, Dict[str, Any]],
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Predict with top contributing features (explainability).
        
        Args:
            features: Feature values
            top_n: Number of top features to return
        
        Returns:
            Prediction dict with 'reasons' field
        """
        # Get prediction
        result = self.predict(features)
        
        # Convert to DataFrame
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features
        
        # Get feature contributions (using feature values * importance as proxy)
        feature_importance = self.model.feature_importances_
        feature_values = features_df[self.feature_names].values[0]
        
        # Calculate contribution scores
        contributions = []
        for i, (name, value, importance) in enumerate(zip(
            self.feature_names, feature_values, feature_importance
        )):
            # Skip zero/null values
            if pd.isna(value) or value == 0:
                continue
            
            # Contribution = value * importance (normalized)
            contribution = abs(value * importance)
            
            contributions.append({
                'feature': name,
                'value': float(value),
                'importance': float(importance),
                'contribution': float(contribution)
            })
        
        # Sort by contribution
        contributions.sort(key=lambda x: x['contribution'], reverse=True)
        
        # Get top N
        top_features = contributions[:top_n]
        
        # Add to result
        result['reasons'] = top_features
        
        # Generate human-readable reasons
        result['reason_codes'] = self._generate_reason_codes(features_df)
        
        return result
    
    def _generate_reason_codes(self, features: pd.DataFrame) -> List[str]:
        """Generate human-readable reason codes"""
        reasons = []
        
        row = features.iloc[0]
        
        # Check various fraud signals
        if row.get('amount_deviation', 0) > 2:
            reasons.append('amount_much_higher_than_usual')
        
        if row.get('tx_count_1h', 0) >= 3:
            reasons.append('high_transaction_velocity')
        
        if row.get('is_night', 0) == 1:
            reasons.append('transaction_at_unusual_hour')
        
        if row.get('device_changed_flag', 0) == 1:
            reasons.append('device_change_detected')
        
        if row.get('location_changed_flag', 0) == 1:
            reasons.append('location_change_detected')
        
        if row.get('is_new_receiver', 0) == 1 and row.get('amount_raw', 0) > 5000:
            reasons.append('large_amount_to_new_receiver')
        
        if row.get('high_amount_at_night', 0) == 1:
            reasons.append('high_amount_at_night')
        
        if row.get('amount_2x_avg', 0) == 1:
            reasons.append('amount_2x_user_average')
        
        if row.get('time_since_last_tx', 0) < 5:  # Less than 5 minutes
            reasons.append('rapid_successive_transaction')
        
        return reasons
    
    def batch_predict(
        self,
        features_df: pd.DataFrame,
        include_reasons: bool = False
    ) -> pd.DataFrame:
        """
        Predict on multiple transactions.
        
        Args:
            features_df: DataFrame with features
            include_reasons: Whether to include reason codes
        
        Returns:
            DataFrame with predictions
        """
        results = []
        
        for idx, row in features_df.iterrows():
            if include_reasons:
                pred = self.predict_with_reasons(row.to_dict())
            else:
                pred = self.predict(row.to_dict())
            
            results.append(pred)
        
        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    # Load predictor
    predictor = FraudPredictor(threshold=0.5)
    
    # Example transaction features
    example_features = {
        'hour': 2,
        'day_of_week': 3,
        'is_weekend': 0,
        'is_night': 1,
        'is_early_morning': 1,
        'is_business_hours': 0,
        'hour_sin': -0.5,
        'hour_cos': 0.87,
        'day_sin': 0.43,
        'day_cos': -0.9,
        'amount_raw': 25000,
        'amount_log': 10.13,
        'amount_very_small': 0,
        'amount_small': 0,
        'amount_medium': 0,
        'amount_large': 0,
        'amount_very_large': 1,
        'tx_count_1h': 5,
        'tx_count_24h': 12,
        'tx_amount_1h': 45000,
        'tx_amount_24h': 78000,
        'tx_amount_1h_log': 10.71,
        'tx_amount_24h_log': 11.26,
        'time_since_last_tx': 3.5,
        'avg_tx_amount_24h': 6500,
        'user_tx_number': 45,
        'user_total_amount': 285000,
        'user_avg_amount': 6333,
        'user_std_amount': 1200,
        'amount_deviation': 15.6,
        'amount_2x_avg': 1,
        'amount_3x_avg': 1,
        'is_new_receiver': 1,
        'receiver_tx_count': 0,
        'device_changed_flag': 1,
        'location_changed_flag': 1,
        'device_or_location_changed': 1,
        'device_and_location_changed': 1,
        'high_amount_at_night': 1,
        'new_receiver_large_amount': 1,
        'device_changed_unusual_amount': 1,
        'high_velocity_large_amount': 1,
        'night_device_change': 1
    }
    
    # Predict
    result = predictor.predict_with_reasons(example_features)
    
    print("Prediction Result:")
    print(f"  Is Fraud: {result['is_fraud']}")
    print(f"  Fraud Score: {result['fraud_score']:.4f}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"\nTop Contributing Features:")
    for reason in result['reasons']:
        print(f"  - {reason['feature']}: {reason['value']:.2f} (importance: {reason['importance']:.4f})")
    print(f"\nReason Codes:")
    for code in result['reason_codes']:
        print(f"  - {code}")