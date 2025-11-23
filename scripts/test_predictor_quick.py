"""Quick predictor test with small sample"""
import sys
import time
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.models.predictor import FraudPredictor

def main():
    print("="*60)
    print("🚀 Quick Predictor Test (Small Sample)")
    print("="*60)
    
    # Load predictor
    print("\n📦 Loading predictor...")
    start = time.time()
    predictor = FraudPredictor()
    print(f"✅ Loaded in {time.time() - start:.2f}s")
    
    # Load SMALL sample
    print("\n📂 Loading SMALL sample (100 transactions)...")
    df = pd.read_csv("data/processed/transactions_features.csv")
    test_df = df.sample(100, random_state=42)
    print(f"✅ Loaded {len(test_df)} transactions")
    
    # Get features
    metadata_cols = ['transaction_id', 'user_id', 'timestamp', 'is_fraud']
    feature_cols = [col for col in test_df.columns if col not in metadata_cols]
    
    print(f"\n   Features: {len(feature_cols)}")
    print(f"   Sample features: {feature_cols[:5]}")
    
    # Test single prediction first
    print("\n🧪 Testing SINGLE prediction...")
    start = time.time()
    single_tx = test_df.iloc[0][feature_cols].to_dict()
    result = predictor.predict(single_tx)
    elapsed_single = time.time() - start
    
    print(f"✅ Single prediction: {elapsed_single*1000:.2f}ms")
    print(f"   Result: {result['fraud_score']:.4f} ({result['risk_level']})")
    
    # Test batch prediction
    print("\n🧪 Testing BATCH prediction (100 transactions)...")
    start = time.time()
    X_test = test_df[feature_cols]
    results = predictor.batch_predict(X_test)
    elapsed_batch = time.time() - start
    
    print(f"✅ Batch prediction: {elapsed_batch*1000:.2f}ms")
    print(f"   Average per tx: {elapsed_batch*10:.2f}ms")
    
    # Estimate for full dataset
    estimated_full = (elapsed_batch / 100) * 100516
    print(f"\n📊 Estimated time for 100K transactions:")
    print(f"   {estimated_full:.1f} seconds ({estimated_full/60:.1f} minutes)")
    
    if estimated_full > 60:
        print(f"\n⚠️  WARNING: This is too slow!")
        print(f"   Expected: <10 seconds")
        print(f"   Estimated: {estimated_full:.1f} seconds")
        print(f"\n   Possible causes:")
        print(f"   1. Feature engineering inside predict() (should be pre-computed)")
        print(f"   2. Inefficient batch_predict() implementation")
        print(f"   3. Large model or too many features")
    else:
        print(f"\n✅ Performance looks good!")

if __name__ == "__main__":
    main()