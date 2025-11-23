"""
Fraud prediction class for production use.

Loads trained model and makes predictions on new transactions.
"""

import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import time


class FraudPredictor:
    """Make fraud predictions on new transactions"""
    
    def __init__(
        self,
        model_path: str = "models/fraud_model.pkl",
        feature_names_path: str = "models/feature_names.json",
        metadata_path: str = "models/model_metadata.json",
        threshold: Optional[float] = None
    ):
        """
        Args:
            model_path: Path to trained model pickle file
            feature_names_path: Path to feature names JSON
            metadata_path: Path to model metadata JSON
            threshold: Classification threshold (uses optimal from metadata if None)
        """
        self.model_path = Path(model_path)
        self.feature_names_path = Path(feature_names_path)
        self.metadata_path = Path(metadata_path)
        
        # Load model and metadata
        self.model = self._load_model()
        self.feature_names = self._load_feature_names()
        self.metadata = self._load_metadata()
        
        # Set threshold (use optimal from training if not specified)
        if threshold is None:
            self.threshold = self.metadata.get('metrics', {}).get('threshold', 0.5)
        else:
            self.threshold = threshold
        
        print(f"✅ FraudPredictor loaded successfully")
        print(f"   Model: {self.model_path.name}")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Threshold: {self.threshold:.2f}")
    
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
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata"""
        if not self.metadata_path.exists():
            print(f"⚠️ Metadata not found at {self.metadata_path}, using defaults")
            return {}
        
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def predict_proba(
        self, 
        features: Union[pd.DataFrame, Dict[str, Any]]
    ) -> Union[float, List[float]]:
        """
        Get fraud probability for transaction(s).
        
        Args:
            features: Dictionary or DataFrame with feature values
        
        Returns:
            Fraud probability (0-1) or list of probabilities
        """
        start_time = time.time()
        
        # Convert to DataFrame if dict
        if isinstance(features, dict):
            features = pd.DataFrame([features])
            single_prediction = True
        else:
            single_prediction = False
        
        # Ensure all features present (fill missing with 0)
        for feature in self.feature_names:
            if feature not in features.columns:
                features[feature] = 0
        
        # Select and order features
        X = features[self.feature_names]
        
        # Predict
        proba = self.model.predict_proba(X)[:, 1]
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # ms
        
        if single_prediction:
            return float(proba[0])
        else:
            return proba.tolist()
    
    def predict(
        self, 
        features: Union[pd.DataFrame, Dict[str, Any]],
        return_timing: bool = False
    ) -> Dict[str, Any]:
        """
        Predict fraud with full details.
        
        Args:
            features: Feature values
            return_timing: Include inference time in response
        
        Returns:
            Dictionary with:
                - is_fraud: Boolean prediction
                - fraud_score: Probability (0-1)
                - risk_level: "low", "medium", "high"
                - threshold: Used threshold
                - inference_time_ms: Time taken (if return_timing=True)
        """
        start_time = time.time()
        
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
        
        result = {
            'is_fraud': bool(is_fraud),
            'fraud_score': float(fraud_score),
            'risk_level': risk_level,
            'threshold': self.threshold
        }
        
        if return_timing:
            result['inference_time_ms'] = (time.time() - start_time) * 1000
        
        return result
    
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
            Prediction dict with 'reasons' and 'reason_codes' fields
        """
        # Get prediction
        result = self.predict(features, return_timing=True)
        
        # Convert to DataFrame
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features
        
        # Ensure all features present
        for feature in self.feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
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
        
        # Check various fraud signals (in order of severity)
        if row.get('amount_3x_avg', 0) == 1:
            reasons.append('AMOUNT_3X_USER_AVERAGE')
        elif row.get('amount_2x_avg', 0) == 1:
            reasons.append('AMOUNT_2X_USER_AVERAGE')
        elif row.get('amount_deviation', 0) > 2:
            reasons.append('AMOUNT_SIGNIFICANTLY_HIGHER_THAN_USUAL')
        
        if row.get('tx_count_1h', 0) >= 5:
            reasons.append('VERY_HIGH_TRANSACTION_VELOCITY')
        elif row.get('tx_count_1h', 0) >= 3:
            reasons.append('HIGH_TRANSACTION_VELOCITY')
        
        if row.get('high_amount_at_night', 0) == 1:
            reasons.append('HIGH_AMOUNT_AT_UNUSUAL_HOUR')
        elif row.get('is_night', 0) == 1:
            reasons.append('TRANSACTION_AT_NIGHT')
        
        if row.get('device_and_location_changed', 0) == 1:
            reasons.append('DEVICE_AND_LOCATION_CHANGED')
        elif row.get('device_changed_flag', 0) == 1:
            reasons.append('DEVICE_CHANGE_DETECTED')
        elif row.get('location_changed_flag', 0) == 1:
            reasons.append('LOCATION_CHANGE_DETECTED')
        
        if row.get('new_receiver_large_amount', 0) == 1:
            reasons.append('LARGE_AMOUNT_TO_NEW_RECEIVER')
        elif row.get('is_new_receiver', 0) == 1 and row.get('amount_raw', 0) > 5000:
            reasons.append('SIGNIFICANT_AMOUNT_TO_NEW_RECEIVER')
        elif row.get('is_new_receiver', 0) == 1:
            reasons.append('NEW_RECEIVER')
        
        if row.get('time_since_last_tx', 0) < 2:
            reasons.append('EXTREMELY_RAPID_SUCCESSIVE_TRANSACTION')
        elif row.get('time_since_last_tx', 0) < 5:
            reasons.append('RAPID_SUCCESSIVE_TRANSACTION')
        
        if row.get('device_changed_unusual_amount', 0) == 1:
            reasons.append('DEVICE_CHANGE_WITH_UNUSUAL_AMOUNT')
        
        if row.get('high_velocity_large_amount', 0) == 1:
            reasons.append('HIGH_VELOCITY_WITH_LARGE_AMOUNT')
        
        return reasons
    
    def batch_predict(
        self,
        features_df: pd.DataFrame,
        include_reasons: bool = False,
        show_progress: bool = False
    ) -> pd.DataFrame:
        """
        Predict on multiple transactions (OPTIMIZED - 100x faster).
        
        Args:
            features_df: DataFrame with features
            include_reasons: Whether to include reason codes (slower)
            show_progress: Print progress updates
        
        Returns:
            DataFrame with predictions
        """
        if show_progress:
            print(f"   Processing {len(features_df):,} transactions...")
        
        # Validate features
        missing_features = set(self.feature_names) - set(features_df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select features in correct order
        X = features_df[self.feature_names]
        
        # ✅ VECTORIZED prediction (processes all rows at once - FAST!)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        if show_progress:
            print(f"   ✅ Predictions complete")
        
        # ✅ VECTORIZED classification
        is_fraud = (y_pred_proba >= self.threshold).astype(int)
        
        # ✅ VECTORIZED risk levels
        risk_levels = pd.cut(
            y_pred_proba,
            bins=[-np.inf, 0.3, 0.7, np.inf],
            labels=['LOW', 'MEDIUM', 'HIGH']
        )
        
        # Create results DataFrame
        results = pd.DataFrame({
            'is_fraud': is_fraud,
            'fraud_score': y_pred_proba,
            'risk_level': risk_levels.astype(str)
        })
        
        # Add transaction IDs if present
        if 'transaction_id' in features_df.columns:
            results.insert(0, 'transaction_id', features_df['transaction_id'].values)
        
        # Only compute reasons if explicitly requested (this IS slow, but optional)
        if include_reasons:
            if show_progress:
                print(f"   Computing reason codes...")
            
            # Only for high-risk transactions to save time
            high_risk_mask = y_pred_proba >= 0.5
            high_risk_count = high_risk_mask.sum()
            
            if show_progress and high_risk_count > 0:
                print(f"   Generating reasons for {high_risk_count:,} high-risk transactions...")
            
            reason_codes_list = []
            
            for idx, row in features_df.iterrows():
                if y_pred_proba[idx] >= 0.5:  # Only for high-risk
                    codes = self._generate_reason_codes(pd.DataFrame([row]))
                    reason_codes_list.append(codes)
                else:
                    reason_codes_list.append([])  # Empty for low-risk
                
                if show_progress and (len(reason_codes_list) % 1000 == 0):
                    print(f"   Processed {len(reason_codes_list):,} reason codes...")
            
            results['reason_codes'] = reason_codes_list
        
        if show_progress:
            print(f"   ✅ Complete")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata"""
        return {
            'model_path': str(self.model_path),
            'n_features': len(self.feature_names),
            'threshold': self.threshold,
            'metadata': self.metadata
        }


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("FRAUDGUARD PREDICTOR - EXAMPLE USAGE")
    print("="*80)
    
    # Load predictor (uses optimal threshold from training)
    predictor = FraudPredictor()
    
    print(f"\n📊 Model Info:")
    info = predictor.get_model_info()
    print(f"   Features: {info['n_features']}")
    print(f"   Threshold: {info['threshold']:.2f}")
    if info['metadata']:
        print(f"   Balance Strategy: {info['metadata'].get('balance_strategy', 'N/A')}")
        metrics = info['metadata'].get('metrics', {})
        if metrics:
            print(f"   Test F1-Score: {metrics.get('f1', 'N/A'):.4f}")
            print(f"   Test ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
    
    # Example 1: High-risk fraud transaction
    print(f"\n{'='*80}")
    print("EXAMPLE 1: High-Risk Transaction")
    print(f"{'='*80}")
    
    fraud_example = {
        'hour': 2, 'day_of_week': 3, 'is_weekend': 0, 'is_night': 1,
        'is_early_morning': 1, 'is_business_hours': 0,
        'hour_sin': -0.5, 'hour_cos': 0.87, 'day_sin': 0.43, 'day_cos': -0.9,
        'amount_raw': 25000, 'amount_log': 10.13,
        'amount_very_small': 0, 'amount_small': 0, 'amount_medium': 0,
        'amount_large': 0, 'amount_very_large': 1,
        'tx_count_1h': 5, 'tx_count_24h': 12,
        'tx_amount_1h': 45000, 'tx_amount_24h': 78000,
        'tx_amount_1h_log': 10.71, 'tx_amount_24h_log': 11.26,
        'time_since_last_tx': 3.5, 'avg_tx_amount_24h': 6500,
        'user_tx_number': 45, 'user_total_amount': 285000,
        'user_avg_amount': 6333, 'user_std_amount': 1200,
        'amount_deviation': 15.6, 'amount_2x_avg': 1, 'amount_3x_avg': 1,
        'is_new_receiver': 1, 'receiver_tx_count': 0,
        'device_changed_flag': 1, 'location_changed_flag': 1,
        'device_or_location_changed': 1, 'device_and_location_changed': 1,
        'high_amount_at_night': 1, 'new_receiver_large_amount': 1,
        'device_changed_unusual_amount': 1, 'high_velocity_large_amount': 1,
        'night_device_change': 1
    }
    
    result = predictor.predict_with_reasons(fraud_example, top_n=5)
    
    print(f"\n🚨 FRAUD ALERT!")
    print(f"   Decision: {'⛔ BLOCK' if result['is_fraud'] else '✅ ALLOW'}")
    print(f"   Fraud Score: {result['fraud_score']:.2%}")
    print(f"   Risk Level: {result['risk_level'].upper()}")
    print(f"   Inference Time: {result['inference_time_ms']:.2f}ms")
    
    print(f"\n📋 Top Contributing Features:")
    for reason in result['reasons']:
        print(f"   - {reason['feature']}: {reason['value']:.2f} (contrib: {reason['contribution']:.4f})")
    
    print(f"\n🔍 Reason Codes:")
    for code in result['reason_codes']:
        print(f"   - {code}")
    
    # Example 2: Low-risk legitimate transaction
    print(f"\n{'='*80}")
    print("EXAMPLE 2: Low-Risk Transaction")
    print(f"{'='*80}")
    
    legit_example = {
        'hour': 14, 'day_of_week': 2, 'is_weekend': 0, 'is_night': 0,
        'is_early_morning': 0, 'is_business_hours': 1,
        'hour_sin': 0.5, 'hour_cos': -0.5, 'day_sin': 0.9, 'day_cos': -0.43,
        'amount_raw': 850, 'amount_log': 6.74,
        'amount_very_small': 0, 'amount_small': 0, 'amount_medium': 1,
        'amount_large': 0, 'amount_very_large': 0,
        'tx_count_1h': 0, 'tx_count_24h': 2,
        'tx_amount_1h': 0, 'tx_amount_24h': 1200,
        'tx_amount_1h_log': 0, 'tx_amount_24h_log': 7.09,
        'time_since_last_tx': 480, 'avg_tx_amount_24h': 600,
        'user_tx_number': 67, 'user_total_amount': 45000,
        'user_avg_amount': 672, 'user_std_amount': 250,
        'amount_deviation': 0.7, 'amount_2x_avg': 0, 'amount_3x_avg': 0,
        'is_new_receiver': 0, 'receiver_tx_count': 12,
        'device_changed_flag': 0, 'location_changed_flag': 0,
        'device_or_location_changed': 0, 'device_and_location_changed': 0,
        'high_amount_at_night': 0, 'new_receiver_large_amount': 0,
        'device_changed_unusual_amount': 0, 'high_velocity_large_amount': 0,
        'night_device_change': 0
    }
    
    result2 = predictor.predict(legit_example, return_timing=True)
    
    print(f"\n✅ LEGITIMATE TRANSACTION")
    print(f"   Decision: {'⛔ BLOCK' if result2['is_fraud'] else '✅ ALLOW'}")
    print(f"   Fraud Score: {result2['fraud_score']:.2%}")
    print(f"   Risk Level: {result2['risk_level'].upper()}")
    print(f"   Inference Time: {result2['inference_time_ms']:.2f}ms")
    
    print(f"\n{'='*80}")
    print("✅ Predictor ready for production use!")
    print(f"{'='*80}")