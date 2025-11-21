"""
Model training pipeline for fraud detection.

Handles:
- Time-based train/test split (no data leakage)
- Class imbalance (SMOTE, undersampling, class weights)
- Time-series cross-validation
- Hyperparameter tuning (RandomizedSearchCV)
- Model evaluation with comprehensive metrics
- SHAP analysis for interpretability
- Model persistence
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import json
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_validate, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap


class FraudDetectionTrainer:
    """Complete training pipeline for fraud detection model"""
    
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        balance_strategy: str = 'class_weight'  # 'smote', 'undersample', 'class_weight'
    ):
        """
        Args:
            test_size: Fraction of data for test set
            random_state: Random seed for reproducibility
            balance_strategy: How to handle class imbalance
                - 'class_weight': Use XGBoost's scale_pos_weight (recommended)
                - 'smote': SMOTE oversampling
                - 'undersample': Random undersampling of majority class
        """
        self.test_size = test_size
        self.random_state = random_state
        self.balance_strategy = balance_strategy
        
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.metrics = {}
        self.best_params = None
        self.shap_values = None
    
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load processed data with features"""
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Parse timestamp if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Separate features and target
        metadata_cols = ['transaction_id', 'user_id']
        target_col = 'is_fraud'
        
        # Keep timestamp for time-based split but not as feature
        feature_cols = [col for col in df.columns 
                       if col not in metadata_cols + [target_col]]
        
        X = df[feature_cols]
        y = df[target_col]
        
        self.feature_names = [col for col in feature_cols if col != 'timestamp']
        
        print(f"Loaded {len(df):,} samples with {len(self.feature_names)} features")
        print(f"Fraud rate: {y.mean():.2%}")
        
        return X, y
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        use_time_split: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split into train/test sets.
        
        Args:
            X: Features (must include 'timestamp' column if use_time_split=True)
            y: Target
            use_time_split: If True, use time-based split (recommended for fraud)
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"\nSplitting data (test_size={self.test_size})...")
        
        if use_time_split and 'timestamp' in X.columns:
            print("Using TIME-BASED split (no data leakage)")
            
            # Sort by timestamp
            sorted_idx = X['timestamp'].sort_values().index
            X_sorted = X.loc[sorted_idx]
            y_sorted = y.loc[sorted_idx]
            
            # Split at time threshold (train on past, test on future)
            split_idx = int(len(X_sorted) * (1 - self.test_size))
            X_train = X_sorted.iloc[:split_idx]
            X_test = X_sorted.iloc[split_idx:]
            y_train = y_sorted.iloc[:split_idx]
            y_test = y_sorted.iloc[split_idx:]
            
            # Drop timestamp from features (not used for modeling)
            X_train = X_train.drop(columns=['timestamp'])
            X_test = X_test.drop(columns=['timestamp'])
            
            print(f"✅ Train: earliest to {X_sorted.iloc[split_idx-1]['timestamp']}")
            print(f"✅ Test:  {X_sorted.iloc[split_idx]['timestamp']} to latest")
            
        else:
            print("⚠️ Using STRATIFIED RANDOM split (not ideal for time-series fraud data)")
            
            # Remove timestamp if present
            X_no_time = X.drop(columns=['timestamp']) if 'timestamp' in X.columns else X
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_no_time, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Train: {len(X_train):,} samples ({y_train.mean():.2%} fraud)")
        print(f"Test:  {len(X_test):,} samples ({y_test.mean():.2%} fraud)")
        
        return X_train, X_test, y_train, y_test
    
    def handle_imbalance(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using specified strategy"""
        
        if self.balance_strategy == 'class_weight':
            print("\n✅ Using CLASS_WEIGHT strategy (XGBoost scale_pos_weight)")
            print("   No resampling - will use scale_pos_weight parameter")
            return X_train, y_train
        
        elif self.balance_strategy == 'smote':
            print("\n✅ Applying SMOTE (Synthetic Minority Over-sampling)...")
            
            smote = SMOTE(random_state=self.random_state)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            print(f"   Before: {len(y_train):,} samples ({y_train.mean():.2%} fraud)")
            print(f"   After:  {len(y_resampled):,} samples ({y_resampled.mean():.2%} fraud)")
            
            return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)
        
        elif self.balance_strategy == 'undersample':
            print("\n✅ Applying Random Undersampling...")
            
            rus = RandomUnderSampler(random_state=self.random_state)
            X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
            
            print(f"   Before: {len(y_train):,} samples ({y_train.mean():.2%} fraud)")
            print(f"   After:  {len(y_resampled):,} samples ({y_resampled.mean():.2%} fraud)")
            
            return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)
        
        else:
            raise ValueError(f"Unknown balance_strategy: {self.balance_strategy}")
    
    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_distributions: Optional[Dict[str, List]] = None,
        n_iter: int = 30,
        cv: int = 5,
        scoring: str = 'f1'
    ) -> Dict[str, Any]:
        """
        Hyperparameter tuning with RandomizedSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_distributions: Parameter grid (uses default if None)
            n_iter: Number of random combinations to try
            cv: Number of cross-validation folds
            scoring: Metric to optimize
        
        Returns:
            Dictionary with best parameters and scores
        """
        print(f"\n{'='*80}")
        print("HYPERPARAMETER TUNING")
        print(f"{'='*80}")
        
        if param_distributions is None:
            param_distributions = {
                'max_depth': [3, 5, 7, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'n_estimators': [100, 200, 300, 500],
                'min_child_weight': [1, 3, 5, 7],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2, 0.3, 0.4],
                'reg_alpha': [0, 0.01, 0.1, 1],
                'reg_lambda': [1, 1.5, 2, 3]
            }
        
        # Base parameters
        base_params = {
            'random_state': self.random_state,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        
        # Add scale_pos_weight if using class_weight strategy
        if self.balance_strategy == 'class_weight':
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            base_params['scale_pos_weight'] = scale_pos_weight
            print(f"scale_pos_weight: {scale_pos_weight:.2f}")
        
        base_model = xgb.XGBClassifier(**base_params)
        
        # Use TimeSeriesSplit for temporal data
        print(f"Using TimeSeriesSplit with {cv} folds (respects temporal order)")
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Random search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            cv=tscv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=2,
            return_train_score=True
        )
        
        print(f"\nSearching {n_iter} random combinations...")
        print("This may take 5-15 minutes depending on data size...\n")
        
        random_search.fit(X_train, y_train)
        
        # Store best model
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        
        print(f"\n{'='*80}")
        print("TUNING RESULTS")
        print(f"{'='*80}")
        print("\n✅ Best Parameters:")
        for param, value in self.best_params.items():
            print(f"   {param:20s}: {value}")
        
        print(f"\n✅ Best CV {scoring.upper()}: {random_search.best_score_:.4f}")
        
        # Show top 5 parameter combinations
        results_df = pd.DataFrame(random_search.cv_results_)
        top_5 = results_df.nsmallest(5, 'rank_test_score')[
            ['rank_test_score', 'mean_test_score', 'std_test_score', 'params']
        ]
        print(f"\nTop 5 Parameter Combinations:")
        print(top_5.to_string(index=False))
        
        return {
            'best_params': self.best_params,
            'best_score': random_search.best_score_,
            'cv_results': random_search.cv_results_,
            'best_estimator': self.model
        }
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None,
        use_tuned_params: bool = True
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            params: Manual parameters (ignored if use_tuned_params=True and tuning was done)
            use_tuned_params: Use parameters from hyperparameter tuning if available
        
        Returns:
            Trained model
        """
        print(f"\n{'='*80}")
        print("MODEL TRAINING")
        print(f"{'='*80}")
        
        # Determine which parameters to use
        if use_tuned_params and self.best_params is not None:
            print("\n✅ Using TUNED parameters from hyperparameter search")
            model_params = self.best_params.copy()
        else:
            print("\n⚠️ Using DEFAULT parameters (consider running tune_hyperparameters first)")
            model_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
            }
            
            if params:
                model_params.update(params)
        
        # Add base parameters
        model_params.update({
            'random_state': self.random_state,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        })
        
        # Calculate scale_pos_weight if using class_weight strategy
        if self.balance_strategy == 'class_weight':
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            model_params['scale_pos_weight'] = scale_pos_weight
        
        print("\nTraining parameters:")
        for param, value in sorted(model_params.items()):
            print(f"  {param:20s}: {value}")
        
        # Train
        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train, verbose=False)
        
        self.model = model
        print("\n✅ Training complete")
        
        return model
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform time-series cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv: Number of CV folds
            params: Model parameters (uses best_params if available)
        
        Returns:
            Cross-validation results
        """
        print(f"\n{'='*80}")
        print(f"CROSS-VALIDATION ({cv} folds)")
        print(f"{'='*80}")
        
        # Setup model parameters
        if self.best_params:
            model_params = self.best_params.copy()
        else:
            model_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': self.random_state,
                'n_jobs': -1
            }
        
        if self.balance_strategy == 'class_weight':
            scale_pos_weight = (y == 0).sum() / (y == 1).sum()
            model_params['scale_pos_weight'] = scale_pos_weight
        
        if params:
            model_params.update(params)
        
        model = xgb.XGBClassifier(**model_params)
        
        # Use TimeSeriesSplit (CRITICAL for fraud detection)
        print("Using TimeSeriesSplit (respects temporal order)")
        cv_splitter = TimeSeriesSplit(n_splits=cv)
        
        # Define scoring metrics
        scoring = {
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'average_precision': 'average_precision'
        }
        
        # Perform CV
        cv_results = cross_validate(
            model, X, y,
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
            verbose=0
        )
        
        # Print results
        print("\nCross-Validation Results:")
        print("="*60)
        for metric in ['precision', 'recall', 'f1', 'roc_auc', 'average_precision']:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            print(f"\n{metric.upper()}:")
            print(f"  Train: {train_scores.mean():.4f} (±{train_scores.std():.4f})")
            print(f"  Test:  {test_scores.mean():.4f} (±{test_scores.std():.4f})")
        
        return cv_results
    
    def evaluate(
        self,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features (uses self.X_test if None)
            y_test: Test labels (uses self.y_test if None)
            threshold: Classification threshold
        
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_test = X_test if X_test is not None else self.X_test
        y_test = y_test if y_test is not None else self.y_test
        
        print(f"\n{'='*80}")
        print(f"TEST SET EVALUATION (threshold={threshold:.2f})")
        print(f"{'='*80}")
        
        # Predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'threshold': threshold,
            'accuracy': (y_pred == y_test).mean(),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'average_precision': average_precision_score(y_test, y_pred_proba),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        })
        
        # Store
        self.metrics = {
            'metrics': metrics,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': cm.tolist()
        }
        
        # Print
        print("\nPerformance Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Precision:          {metrics['precision']:.4f}")
        print(f"  Recall:             {metrics['recall']:.4f}")
        print(f"  F1-Score:           {metrics['f1']:.4f}")
        print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
        print(f"  Average Precision:  {metrics['average_precision']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"               Legit  Fraud")
        print(f"  Actual Legit  {tn:5d}  {fp:5d}")
        print(f"         Fraud  {fn:5d}  {tp:5d}")
        
        print(f"\nFraud Detection Performance:")
        print(f"  Fraud Caught:     {tp:,} / {tp+fn:,} ({metrics['recall']:.1%})")
        print(f"  False Alarms:     {fp:,} / {fp+tn:,} ({fp/(fp+tn):.1%})")
        print(f"  Missed Fraud:     {fn:,} (cost: fraud losses)")
        print(f"  Correct Blocks:   {tp:,} (savings)")
        
        return metrics
    
    def find_optimal_threshold(
        self,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        metric: str = 'f1',
        plot: bool = True
    ) -> float:
        """
        Find optimal classification threshold.
        
        Args:
            X_test: Test features
            y_test: Test labels
            metric: Metric to optimize ('f1', 'precision', 'recall')
            plot: Whether to plot results
        
        Returns:
            Optimal threshold
        """
        X_test = X_test if X_test is not None else self.X_test
        y_test = y_test if y_test is not None else self.y_test
        
        print(f"\n{'='*80}")
        print(f"THRESHOLD OPTIMIZATION (maximizing {metric.upper()})")
        print(f"{'='*80}")
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        thresholds = np.arange(0.1, 0.95, 0.05)
        scores = []
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            p = precision_score(y_test, y_pred, zero_division=0)
            r = recall_score(y_test, y_pred, zero_division=0)
            f = f1_score(y_test, y_pred, zero_division=0)
            
            precisions.append(p)
            recalls.append(r)
            
            if metric == 'f1':
                scores.append(f)
            elif metric == 'precision':
                scores.append(p)
            elif metric == 'recall':
                scores.append(r)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        print(f"\n✅ Optimal threshold: {optimal_threshold:.2f}")
        print(f"   {metric.upper()}: {optimal_score:.4f}")
        print(f"   Precision: {precisions[optimal_idx]:.4f}")
        print(f"   Recall: {recalls[optimal_idx]:.4f}")
        
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(thresholds, precisions, 'b-', marker='o', label='Precision', linewidth=2)
            plt.plot(thresholds, recalls, 'r-', marker='s', label='Recall', linewidth=2)
            plt.plot(thresholds, scores, 'g-', marker='^', label=metric.upper(), linewidth=2)
            plt.axvline(x=optimal_threshold, color='purple', linestyle='--', 
                       label=f'Optimal: {optimal_threshold:.2f}', linewidth=2)
            plt.xlabel('Threshold', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.title(f'Metrics vs Threshold (Optimizing {metric.upper()})', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'docs/optimal_threshold_{metric}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   Plot saved to: docs/optimal_threshold_{metric}.png")
        
        return optimal_threshold
    
    def analyze_shap(
        self,
        X_test: Optional[pd.DataFrame] = None,
        output_dir: str = "docs",
        max_display: int = 20,
        save_values: bool = True
    ) -> pd.DataFrame:
        """
        Generate SHAP explanations for model interpretability.
        
        Args:
            X_test: Test features (uses self.X_test if None)
            output_dir: Directory to save plots
            max_display: Number of features to display
            save_values: Whether to save SHAP values
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        X_test = X_test if X_test is not None else self.X_test
        
        print(f"\n{'='*80}")
        print("SHAP ANALYSIS (Model Interpretability)")
        print(f"{'='*80}")
        print("\nGenerating SHAP explanations (this may take 2-3 minutes)...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create explainer
        explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values (use sample if dataset is large)
        if len(X_test) > 1000:
            print(f"Using sample of 1000 transactions (dataset has {len(X_test):,})")
            sample_idx = np.random.choice(len(X_test), 1000, replace=False)
            X_sample = X_test.iloc[sample_idx]
            shap_values = explainer.shap_values(X_sample)
        else:
            X_sample = X_test
            shap_values = explainer.shap_values(X_test)
        
        self.shap_values = shap_values
        
        # 1. Summary Plot (beeswarm)
        print("\n1. Creating SHAP summary plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, 
                         feature_names=self.feature_names, 
                         show=False, max_display=max_display)
        plt.tight_layout()
        plt.savefig(output_path / 'shap_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved to: {output_path / 'shap_summary.png'}")
        
        # 2. Bar Plot (mean absolute SHAP values)
        print("2. Creating SHAP importance bar plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type='bar',
                         feature_names=self.feature_names, 
                         show=False, max_display=max_display)
        plt.tight_layout()
        plt.savefig(output_path / 'shap_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved to: {output_path / 'shap_importance.png'}")
        
        # 3. Waterfall plot for a fraud example
        if self.y_test is not None:
            fraud_indices = np.where(self.y_test == 1)[0]
            if len(fraud_indices) > 0:
                print("3. Creating waterfall plot for fraud example...")
                fraud_idx = fraud_indices[0]
                
                shap_exp = shap.Explanation(
                    values=shap_values[fraud_idx],
                    base_values=explainer.expected_value,
                    data=X_sample.iloc[fraud_idx],
                    feature_names=self.feature_names
                )
                
                plt.figure(figsize=(10, 8))
                shap.waterfall_plot(shap_exp, show=False, max_display=max_display)
                plt.tight_layout()
                plt.savefig(output_path / 'shap_waterfall_fraud.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   ✅ Saved to: {output_path / 'shap_waterfall_fraud.png'}")
        
        # Calculate feature importance
        shap_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('shap_importance', ascending=False)
        
        print(f"\n{'='*60}")
        print("Top 15 Most Important Features (SHAP):")
        print("="*60)
        for i, row in shap_importance.head(15).iterrows():
            print(f"{i+1:2d}. {row['feature']:30s} {row['shap_importance']:.4f}")
        
        # Compare with XGBoost built-in importance
        xgb_importance = pd.DataFrame({
            'feature': self.feature_names,
            'xgb_importance': self.model.feature_importances_
        }).sort_values('xgb_importance', ascending=False)
        
        comparison = shap_importance.merge(xgb_importance, on='feature')
        
        print(f"\n{'='*60}")
        print("Feature Importance Comparison (Top 10):")
        print("="*60)
        print(f"{'Feature':<30s} {'SHAP':>10s} {'XGBoost':>10s}")
        print("-"*60)
        for _, row in comparison.head(10).iterrows():
            print(f"{row['feature']:<30s} {row['shap_importance']:>10.4f} {row['xgb_importance']:>10.4f}")
        
        # Save if requested
        if save_values:
            shap_file = output_path / 'shap_importance.csv'
            shap_importance.to_csv(shap_file, index=False)
            print(f"\n✅ SHAP importance saved to: {shap_file}")
        
        return shap_importance
    
    def plot_evaluation(
        self, 
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        output_dir: str = "docs"
    ):
        """
        Generate comprehensive evaluation plots.
        
        Args:
            X_test: Test features
            y_test: Test labels
            output_dir: Directory to save plots
        """
        X_test = X_test if X_test is not None else self.X_test
        y_test = y_test if y_test is not None else self.y_test
        
        if self.model is None or X_test is None or y_test is None:
            raise ValueError("Model not trained or test data not available")
        
        print(f"\n{'='*80}")
        print("GENERATING EVALUATION PLOTS")
        print(f"{'='*80}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        print("1. Confusion matrix...")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('True Label', fontsize=12)
        axes[0, 0].set_xlabel('Predicted Label', fontsize=12)
        
        # 2. ROC Curve
        print("2. ROC curve...")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        axes[0, 1].plot(fpr, tpr, linewidth=2.5, label=f'ROC (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
        axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
        axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
        axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        print("3. Precision-Recall curve...")
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        axes[1, 0].plot(recall, precision, linewidth=2.5, 
                       label=f'PR (AP = {avg_precision:.3f})')
        axes[1, 0].set_xlabel('Recall', fontsize=12)
        axes[1, 0].set_ylabel('Precision', fontsize=12)
        axes[1, 0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction Distribution
        print("4. Prediction distribution...")
        fraud_scores = y_pred_proba[y_test == 1]
        legit_scores = y_pred_proba[y_test == 0]
        
        axes[1, 1].hist(legit_scores, bins=50, alpha=0.6, 
                       label='Legitimate', density=True, color='green')
        axes[1, 1].hist(fraud_scores, bins=50, alpha=0.6, 
                       label='Fraud', density=True, color='red')
        axes[1, 1].axvline(x=0.5, color='purple', linestyle='--', 
                          linewidth=2, label='Threshold (0.5)')
        axes[1, 1].set_xlabel('Prediction Score', fontsize=12)
        axes[1, 1].set_ylabel('Density', fontsize=12)
        axes[1, 1].set_title('Prediction Score Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        eval_plot_path = output_path / 'model_evaluation.png'
        plt.savefig(eval_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✅ Evaluation plots saved to: {eval_plot_path}")
        
        # Feature importance plot
        self.plot_feature_importance(output_dir)
    
    def plot_feature_importance(
        self, 
        output_dir: str = "docs", 
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Plot XGBoost built-in feature importance.
        
        Args:
            output_dir: Directory to save plot
            top_n: Number of top features to display
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        output_path = Path(output_dir)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot top N
        print(f"5. Feature importance (top {top_n})...")
        plt.figure(figsize=(10, 12))
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'].values, fontsize=10)
        plt.xlabel('Importance (Gain)', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances (XGBoost)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        importance_plot_path = output_path / 'feature_importance.png'
        plt.savefig(importance_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved to: {importance_plot_path}")
        
        return importance_df
    
    def save_model(
        self, 
        model_dir: str = "models",
        save_shap: bool = False
    ):
        """
        Save trained model and metadata.
        
        Args:
            model_dir: Directory to save model
            save_shap: Whether to save SHAP values (can be large)
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        print(f"\n{'='*80}")
        print("SAVING MODEL ARTIFACTS")
        print(f"{'='*80}")
        
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        # 1. Save model
        model_file = model_path / "fraud_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\n✅ Model saved to: {model_file}")
        
        # 2. Save feature names
        features_file = model_path / "feature_names.json"
        with open(features_file, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"✅ Feature names saved to: {features_file}")
        
        # 3. Save best parameters
        if self.best_params:
            params_file = model_path / "best_params.json"
            with open(params_file, 'w') as f:
                json.dump(self.best_params, f, indent=2)
            print(f"✅ Best parameters saved to: {params_file}")
        
        # 4. Save metrics
        if self.metrics:
            metrics_file = model_path / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            print(f"✅ Metrics saved to: {metrics_file}")
        
        # 5. Save complete metadata
        metadata = {
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'balance_strategy': self.balance_strategy,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'metrics': self.metrics.get('metrics', {}),
            'train_fraud_rate': float(self.y_train.mean()) if self.y_train is not None else None,
            'test_fraud_rate': float(self.y_test.mean()) if self.y_test is not None else None,
            'n_train_samples': len(self.y_train) if self.y_train is not None else None,
            'n_test_samples': len(self.y_test) if self.y_test is not None else None,
        }
        
        metadata_file = model_path / "model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Complete metadata saved to: {metadata_file}")
        
        # 6. Optionally save SHAP values
        if save_shap and self.shap_values is not None:
            shap_file = model_path / "shap_values.pkl"
            with open(shap_file, 'wb') as f:
                pickle.dump(self.shap_values, f)
            print(f"✅ SHAP values saved to: {shap_file}")
        
        print(f"\n{'='*80}")
        print("ALL ARTIFACTS SAVED SUCCESSFULLY")
        print(f"{'='*80}")


def train_model(
    data_path: str = "data/processed/transactions_features.csv",
    model_dir: str = "models",
    balance_strategy: str = 'class_weight',
    tune_hyperparams: bool = True,
    n_iter: int = 30,
    cv: int = 5,
    run_shap: bool = True
) -> FraudDetectionTrainer:
    """
    Complete end-to-end training pipeline (convenience function).
    
    Args:
        data_path: Path to processed features
        model_dir: Directory to save model
        balance_strategy: Imbalance handling ('class_weight', 'smote', 'undersample')
        tune_hyperparams: Whether to run hyperparameter tuning
        n_iter: Number of random search iterations
        cv: Number of cross-validation folds
        run_shap: Whether to run SHAP analysis
    
    Returns:
        Trained FraudDetectionTrainer instance
    """
    print("="*80)
    print("FRAUDGUARD MODEL TRAINING PIPELINE")
    print("="*80)
    
    # Initialize trainer
    trainer = FraudDetectionTrainer(
        test_size=0.2,
        random_state=42,
        balance_strategy=balance_strategy
    )
    
    # 1. Load data
    X, y = trainer.load_data(data_path)
    
    # 2. Split (time-based)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, use_time_split=True)
    
    # 3. Handle imbalance
    X_train_balanced, y_train_balanced = trainer.handle_imbalance(X_train, y_train)
    
    # 4. Hyperparameter tuning (optional but recommended)
    if tune_hyperparams:
        tuning_results = trainer.tune_hyperparameters(
            X_train_balanced, 
            y_train_balanced,
            n_iter=n_iter,
            cv=cv
        )
        # Model is already set to best estimator
    else:
        # Train with default parameters
        trainer.train(X_train_balanced, y_train_balanced, use_tuned_params=False)
    
    # 5. Cross-validation (on full training set with best params)
    print("\n" + "="*80)
    print("FINAL CROSS-VALIDATION WITH BEST PARAMETERS")
    print("="*80)
    trainer.cross_validate(X_train_balanced, y_train_balanced, cv=cv)
    
    # 6. Evaluate on test set
    trainer.evaluate(X_test, y_test, threshold=0.5)
    
    # 7. Find optimal threshold
    optimal_threshold = trainer.find_optimal_threshold(X_test, y_test, metric='f1', plot=True)
    
    # 8. Re-evaluate with optimal threshold
    print("\n" + "="*80)
    print("FINAL EVALUATION WITH OPTIMAL THRESHOLD")
    print("="*80)
    trainer.evaluate(X_test, y_test, threshold=optimal_threshold)
    
    # 9. Generate plots
    trainer.plot_evaluation(X_test, y_test)
    
    # 10. SHAP analysis (optional but recommended)
    if run_shap:
        trainer.analyze_shap(X_test)
    
    # 11. Save everything
    trainer.save_model(model_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETE")
    print("="*80)
    
    final_metrics = trainer.metrics['metrics']
    print(f"\n📊 Final Performance:")
    print(f"  Precision:          {final_metrics['precision']:.4f}")
    print(f"  Recall:             {final_metrics['recall']:.4f}")
    print(f"  F1-Score:           {final_metrics['f1']:.4f}")
    print(f"  ROC-AUC:            {final_metrics['roc_auc']:.4f}")
    print(f"  Optimal Threshold:  {final_metrics['threshold']:.2f}")
    
    print(f"\n📁 Artifacts saved to:")
    print(f"  Model:              {model_dir}/fraud_model.pkl")
    print(f"  Metadata:           {model_dir}/model_metadata.json")
    print(f"  Plots:              docs/*.png")
    
    print("\n🚀 Ready for deployment!")
    print("   Next: Create predictor API (src/models/predictor.py)")
    
    return trainer


# Example usage
if __name__ == "__main__":
    # Run complete training pipeline
    trainer = train_model(
        data_path="data/processed/transactions_features.csv",
        model_dir="models",
        balance_strategy='class_weight',  # Recommended for fraud
        tune_hyperparams=True,             # Enable hyperparameter tuning
        n_iter=30,                         # Try 30 random combinations
        cv=5,                              # 5-fold time-series CV
        run_shap=True                      # Enable SHAP analysis
    )
    
    print("\n✅ Training complete! Model ready for use.")