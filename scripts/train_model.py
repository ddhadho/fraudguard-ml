"""
Script to train fraud detection model.

Usage:
    # Basic training
    python scripts/train_model.py
    
    # With hyperparameter tuning
    python scripts/train_model.py --tune --n-iter 50
    
    # Different balance strategy
    python scripts/train_model.py --balance-strategy smote
    
    # Skip SHAP (faster)
    python scripts/train_model.py --no-shap
    
    # More CV folds
    python scripts/train_model.py --cv 10
"""
import argparse
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.trainer import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Train fraud detection model with XGBoost",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training (no tuning)
  python scripts/train_model.py --no-tune
  
  # Full pipeline with tuning
  python scripts/train_model.py --tune --n-iter 30
  
  # Use SMOTE for imbalance
  python scripts/train_model.py --balance-strategy smote
  
  # Skip SHAP analysis (saves time)
  python scripts/train_model.py --no-shap
        """
    )
    
    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/transactions_features.csv",
        help="Path to processed features CSV (default: data/processed/transactions_features.csv)"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save trained model (default: models)"
    )
    
    # Training arguments
    parser.add_argument(
        "--balance-strategy",
        type=str,
        default="class_weight",
        choices=['class_weight', 'smote', 'undersample'],
        help="Class imbalance handling strategy (default: class_weight)"
    )
    
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    
    # Hyperparameter tuning arguments
    parser.add_argument(
        "--tune",
        action="store_true",
        default=True,
        help="Enable hyperparameter tuning (default: True)"
    )
    
    parser.add_argument(
        "--no-tune",
        dest="tune",
        action="store_false",
        help="Disable hyperparameter tuning (faster but less optimal)"
    )
    
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Number of random search iterations for tuning (default: 30)"
    )
    
    # SHAP analysis arguments
    parser.add_argument(
        "--shap",
        action="store_true",
        default=True,
        help="Run SHAP analysis (default: True)"
    )
    
    parser.add_argument(
        "--no-shap",
        dest="shap",
        action="store_false",
        help="Skip SHAP analysis (saves 2-3 minutes)"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*80)
    print("FRAUDGUARD MODEL TRAINING")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Data:              {args.data}")
    print(f"  Model directory:   {args.model_dir}")
    print(f"  Balance strategy:  {args.balance_strategy}")
    print(f"  Cross-validation:  {args.cv} folds")
    print(f"  Hyperparameter tuning: {'Yes' if args.tune else 'No'}")
    if args.tune:
        print(f"    Iterations:      {args.n_iter}")
    print(f"  SHAP analysis:     {'Yes' if args.shap else 'No'}")
    print("="*80)
    
    # Confirm data exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"\n❌ Error: Data file not found: {args.data}")
        print("   Run feature engineering first:")
        print("   python scripts/engineer_features.py")
        sys.exit(1)
    
    # Start training
    start_time = time.time()
    
    try:
        trainer = train_model(
            data_path=args.data,
            model_dir=args.model_dir,
            balance_strategy=args.balance_strategy,
            tune_hyperparams=args.tune,
            n_iter=args.n_iter,
            cv=args.cv,
            run_shap=args.shap
        )
        
        # Training complete
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"\nTotal time: {elapsed/60:.1f} minutes")
        
        # Show final metrics
        if trainer.metrics:
            metrics = trainer.metrics.get('metrics', {})
            print(f"\n📊 Final Performance:")
            print(f"  Precision:  {metrics.get('precision', 0):.4f}")
            print(f"  Recall:     {metrics.get('recall', 0):.4f}")
            print(f"  F1-Score:   {metrics.get('f1', 0):.4f}")
            print(f"  ROC-AUC:    {metrics.get('roc_auc', 0):.4f}")
            print(f"  Threshold:  {metrics.get('threshold', 0.5):.2f}")
        
        print(f"\n📁 Model saved to: {args.model_dir}/")
        print(f"📈 Plots saved to: docs/")
        
        print("\n🚀 Next steps:")
        print("  1. Review plots in docs/ folder")
        print("  2. Test predictor: python -c \"from src.models.predictor import FraudPredictor; p = FraudPredictor(); print('Model loaded!')\"")
        print("  3. Build API: Create FastAPI endpoint")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure you've run feature engineering first:")
        print("   python scripts/engineer_features.py")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()