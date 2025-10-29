"""
Script to engineer features from raw transaction data.

Usage:
    python scripts/engineer_features.py
    python scripts/engineer_features.py --input data/raw/transactions.csv
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.features.engineering import engineer_features


def main():
    parser = argparse.ArgumentParser(description="Engineer features from transaction data")
    
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/transactions.csv",
        help="Input CSV path (default: data/raw/transactions.csv)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/transactions_features.csv",
        help="Output CSV path (default: data/processed/transactions_features.csv)"
    )
    
    args = parser.parse_args()
    
    # Engineer features
    engineer_features(
        input_path=args.input,
        output_path=args.output
    )


if __name__ == "__main__":
    main()