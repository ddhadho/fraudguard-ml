"""
Test the fraud predictor on real generated data.
Usage:
    python scripts/test_predictor.py
"""
import sys
import time
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.models.predictor import FraudPredictor


def main():
    print("="*60)
    print("🚀 FraudGuard Predictor Validation")
    print("="*60)

    # Load predictor
    print("\n📦 Loading predictor...")
    try:
        predictor = FraudPredictor()
        print("✅ Predictor loaded")
    except Exception as e:
        print(f"❌ Failed to load predictor: {e}")
        return

    # Load existing processed feature data
    print("\n📂 Loading test data from existing features CSV...")
    try:
        df = pd.read_csv("data/processed/transactions_features.csv")
        print(f"✅ Loaded {len(df)} transactions")
        print(f"   Fraud rate: {df['is_fraud'].mean():.2%}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Separate metadata vs features
    metadata_cols = ['transaction_id', 'user_id', 'timestamp', 'is_fraud']
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    # Run batch predictions
    print(f"\n🔄 Running predictions on {len(df)} transactions...")
    start_time = time.time()
    try:
        results = predictor.batch_predict(df[feature_cols])
        elapsed = time.time() - start_time
        print(f"✅ Completed in {elapsed*1000:.2f}ms")
        print(f"   Average: {elapsed*1000/len(df):.2f}ms per transaction")

        # Merge predictions with actual labels
        df['predicted_fraud'] = results['is_fraud'].values
        df['fraud_score'] = results['fraud_score'].values

        # Calculate metrics
        tp = ((df['is_fraud'] == 1) & (df['predicted_fraud'] == 1)).sum()
        tn = ((df['is_fraud'] == 0) & (df['predicted_fraud'] == 0)).sum()
        fp = ((df['is_fraud'] == 0) & (df['predicted_fraud'] == 1)).sum()
        fn = ((df['is_fraud'] == 1) & (df['predicted_fraud'] == 0)).sum()

        accuracy = (tp + tn) / len(df)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print("\n🎯 Classification Metrics:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")

        print("\n📈 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Legit  Fraud")
        print(f"   Actual Legit  {tn:5d}  {fp:5d}")
        print(f"   Actual Fraud  {fn:5d}  {tp:5d}")

        # Business impact
        fraud_caught_pct = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        false_alarm_pct = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0

        print(f"\n💰 Business Impact:")
        print(f"   Fraud Caught: {tp}/{tp+fn} ({fraud_caught_pct:.1f}%)")
        print(f"   False Alarms: {fp}/{fp+tn} ({false_alarm_pct:.1f}%)")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()  
