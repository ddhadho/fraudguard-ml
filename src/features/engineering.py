"""
Feature engineering for fraud detection.

Extracts features from raw transaction data that capture fraud signals.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import timedelta


class FeatureEngineer:
    """Extract features from transaction data"""
    
    def __init__(self, lookback_hours: int = 24):
        """
        Args:
            lookback_hours: How many hours to look back for velocity features
        """
        self.lookback_hours = lookback_hours
        self.feature_names = []
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features from transaction dataframe.
        
        Args:
            df: DataFrame with columns:
                - transaction_id
                - user_id
                - timestamp
                - amount
                - receiver
                - device_id
                - location
                - device_changed
                - location_changed
                - is_fraud (target)
        
        Returns:
            DataFrame with extracted features
        """
        print("Extracting features...")
        
        # Make a copy
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by user and time (important for velocity features)
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        # Extract features
        df = self._add_time_features(df)
        df = self._add_amount_features(df)
        df = self._add_velocity_features(df)
        df = self._add_user_history_features(df)
        df = self._add_receiver_features(df)
        df = self._add_device_location_features(df)
        df = self._add_interaction_features(df)
        
        print(f"✅ Extracted {len(self.feature_names)} features")
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features"""
        print("  - Time features...")
        
        # Hour of day (0-23)
        df['hour'] = df['timestamp'].dt.hour
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Time buckets
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['is_early_morning'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Cyclical encoding (hour is cyclical: 23 and 0 are close)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week cyclical
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        self.feature_names.extend([
            'hour', 'day_of_week', 'is_weekend', 'is_night', 
            'is_early_morning', 'is_business_hours',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ])
        
        return df
    
    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Amount-based features"""
        print("  - Amount features...")
        
        # Raw amount
        df['amount_raw'] = df['amount']
        
        # Log amount (to handle skewed distribution)
        df['amount_log'] = np.log1p(df['amount'])
        
        # Amount bins
        df['amount_very_small'] = (df['amount'] < 100).astype(int)
        df['amount_small'] = ((df['amount'] >= 100) & (df['amount'] < 1000)).astype(int)
        df['amount_medium'] = ((df['amount'] >= 1000) & (df['amount'] < 5000)).astype(int)
        df['amount_large'] = ((df['amount'] >= 5000) & (df['amount'] < 20000)).astype(int)
        df['amount_very_large'] = (df['amount'] >= 20000).astype(int)
        
        self.feature_names.extend([
            'amount_raw', 'amount_log',
            'amount_very_small', 'amount_small', 'amount_medium',
            'amount_large', 'amount_very_large'
        ])
        
        return df
    
    def _add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transaction velocity features (frequency in time windows)"""
        print("  - Velocity features...")
        
        # For each transaction, count how many transactions the user had in past X hours
        # This is computationally expensive, so we'll use a rolling window approach
        
        # Convert lookback to timedelta
        lookback = timedelta(hours=self.lookback_hours)
        
        # Group by user
        grouped = df.groupby('user_id')
        
        # Initialize velocity features
        df['tx_count_1h'] = 0
        df['tx_count_24h'] = 0
        df['tx_amount_1h'] = 0.0
        df['tx_amount_24h'] = 0.0
        df['time_since_last_tx'] = 0.0
        
        # Calculate for each user
        for user_id, group in grouped:
            indices = group.index
            timestamps = group['timestamp'].values
            amounts = group['amount'].values
            
            for i, idx in enumerate(indices):
                current_time = timestamps[i]
                
                # Look back at previous transactions
                if i > 0:
                    # Time since last transaction (in minutes)
                    time_diff = (current_time - timestamps[i-1]).astype('timedelta64[m]').astype(float)
                    df.loc[idx, 'time_since_last_tx'] = time_diff
                    
                    # Count transactions in last 1 hour
                    one_hour_ago = current_time - pd.Timedelta(hours=1)
                    mask_1h = (timestamps[:i] >= one_hour_ago)
                    df.loc[idx, 'tx_count_1h'] = mask_1h.sum()
                    df.loc[idx, 'tx_amount_1h'] = amounts[:i][mask_1h].sum()
                    
                    # Count transactions in last 24 hours
                    one_day_ago = current_time - pd.Timedelta(hours=24)
                    mask_24h = (timestamps[:i] >= one_day_ago)
                    df.loc[idx, 'tx_count_24h'] = mask_24h.sum()
                    df.loc[idx, 'tx_amount_24h'] = amounts[:i][mask_24h].sum()
        
        # Derived velocity features
        df['tx_amount_1h_log'] = np.log1p(df['tx_amount_1h'])
        df['tx_amount_24h_log'] = np.log1p(df['tx_amount_24h'])
        
        # Average transaction amount in window
        df['avg_tx_amount_24h'] = np.where(
            df['tx_count_24h'] > 0,
            df['tx_amount_24h'] / df['tx_count_24h'],
            0
        )
        
        self.feature_names.extend([
            'tx_count_1h', 'tx_count_24h',
            'tx_amount_1h', 'tx_amount_24h',
            'tx_amount_1h_log', 'tx_amount_24h_log',
            'time_since_last_tx', 'avg_tx_amount_24h'
        ])
        
        return df
    
    def _add_user_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """User historical behavior features"""
        print("  - User history features...")
        
        # Calculate cumulative statistics for each user
        grouped = df.groupby('user_id')
        
        # Transaction number (1st, 2nd, 3rd, ...)
        df['user_tx_number'] = grouped.cumcount() + 1
        
        # Cumulative statistics (expanding window)
        df['user_total_amount'] = grouped['amount'].cumsum()
        df['user_avg_amount'] = grouped['amount'].expanding().mean().reset_index(level=0, drop=True)
        df['user_std_amount'] = grouped['amount'].expanding().std().reset_index(level=0, drop=True)
        
        # Fill NaN std (first transaction has no std)
        df['user_std_amount'] = df['user_std_amount'].fillna(0)
        
        # Amount deviation from user's average
        df['amount_deviation'] = np.where(
            df['user_avg_amount'] > 0,
            (df['amount'] - df['user_avg_amount']) / (df['user_std_amount'] + 1),
            0
        )
        
        # Is this amount much higher than usual?
        df['amount_2x_avg'] = (df['amount'] > 2 * df['user_avg_amount']).astype(int)
        df['amount_3x_avg'] = (df['amount'] > 3 * df['user_avg_amount']).astype(int)
        
        self.feature_names.extend([
            'user_tx_number', 'user_total_amount',
            'user_avg_amount', 'user_std_amount',
            'amount_deviation', 'amount_2x_avg', 'amount_3x_avg'
        ])
        
        return df
    
    def _add_receiver_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Receiver-related features"""
        print("  - Receiver features...")
        
        # Group by user
        grouped = df.groupby('user_id')
        
        # Has user sent to this receiver before?
        df['is_new_receiver'] = 0
        df['receiver_tx_count'] = 0
        
        for user_id, group in grouped:
            indices = group.index
            receivers = group['receiver'].values
            
            seen_receivers = set()
            receiver_counts = {}
            
            for i, idx in enumerate(indices):
                receiver = receivers[i]
                
                # Is this a new receiver?
                if receiver not in seen_receivers:
                    df.loc[idx, 'is_new_receiver'] = 1
                    seen_receivers.add(receiver)
                
                # How many times have we sent to this receiver?
                df.loc[idx, 'receiver_tx_count'] = receiver_counts.get(receiver, 0)
                receiver_counts[receiver] = receiver_counts.get(receiver, 0) + 1
        
        self.feature_names.extend([
            'is_new_receiver', 'receiver_tx_count'
        ])
        
        return df
    
    def _add_device_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Device and location features"""
        print("  - Device/location features...")
        
        # Binary indicators (already in data)
        df['device_changed_flag'] = df['device_changed'].astype(int)
        df['location_changed_flag'] = df['location_changed'].astype(int)
        
        # Combined flag
        df['device_or_location_changed'] = (
            (df['device_changed']) | (df['location_changed'])
        ).astype(int)
        
        df['device_and_location_changed'] = (
            (df['device_changed']) & (df['location_changed'])
        ).astype(int)
        
        self.feature_names.extend([
            'device_changed_flag', 'location_changed_flag',
            'device_or_location_changed', 'device_and_location_changed'
        ])
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interaction features (combinations of other features)"""
        print("  - Interaction features...")
        
        # High amount + night time
        df['high_amount_at_night'] = (
            (df['amount'] > df['user_avg_amount'] * 2) & 
            (df['is_night'] == 1)
        ).astype(int)
        
        # New receiver + large amount
        df['new_receiver_large_amount'] = (
            (df['is_new_receiver'] == 1) & 
            (df['amount'] > 5000)
        ).astype(int)
        
        # Device changed + unusual amount
        df['device_changed_unusual_amount'] = (
            (df['device_changed_flag'] == 1) & 
            (df['amount_2x_avg'] == 1)
        ).astype(int)
        
        # High velocity + large amount
        df['high_velocity_large_amount'] = (
            (df['tx_count_1h'] >= 3) & 
            (df['amount'] > 3000)
        ).astype(int)
        
        # Night transaction + device change
        df['night_device_change'] = (
            (df['is_night'] == 1) & 
            (df['device_changed_flag'] == 1)
        ).astype(int)
        
        self.feature_names.extend([
            'high_amount_at_night', 'new_receiver_large_amount',
            'device_changed_unusual_amount', 'high_velocity_large_amount',
            'night_device_change'
        ])
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return self.feature_names
    
    def get_feature_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get only the feature columns (excluding metadata).
        
        Returns DataFrame ready for ML model.
        """
        # Features to include
        feature_cols = self.feature_names
        
        # Target
        target_col = 'is_fraud'
        
        # Metadata to keep
        metadata_cols = ['transaction_id', 'user_id', 'timestamp']
        
        # Select columns
        cols_to_keep = metadata_cols + feature_cols + [target_col]
        
        return df[cols_to_keep]


def engineer_features(
    input_path: str = "data/raw/transactions.csv",
    output_path: str = "data/processed/transactions_features.csv"
) -> pd.DataFrame:
    """
    Convenience function to load data, engineer features, and save.
    
    Args:
        input_path: Path to raw transaction CSV
        output_path: Path to save engineered features
    
    Returns:
        DataFrame with features
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} transactions\n")
    
    # Engineer features
    engineer = FeatureEngineer(lookback_hours=24)
    df_features = engineer.fit_transform(df)
    
    # Get only feature columns
    df_ml_ready = engineer.get_feature_dataframe(df_features)
    
    # Save
    df_ml_ready.to_csv(output_path, index=False)
    print(f"\n✅ Features saved to: {output_path}")
    print(f"Shape: {df_ml_ready.shape}")
    print(f"Features: {len(engineer.get_feature_names())}")
    
    return df_ml_ready