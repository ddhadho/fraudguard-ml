"""
Script to generate synthetic transaction data.

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --num-transactions 50000
    python scripts/generate_data.py --fraud-rate 0.05
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.generator import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data")
    
    parser.add_argument(
        "--num-users",
        type=int,
        default=1000,
        help="Number of unique users (default: 1000)"
    )
    
    parser.add_argument(
        "--num-transactions",
        type=int,
        default=100000,
        help="Total number of transactions (default: 100000)"
    )
    
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.02,
        help="Fraud rate as decimal (default: 0.02 = 2%%)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/transactions.csv",
        help="Output CSV path (default: data/raw/transactions.csv)"
    )
    
    args = parser.parse_args()
    
    # Generate dataset
    generate_dataset(
        output_path=args.output,
        num_users=args.num_users,
        num_transactions=args.num_transactions,
        fraud_rate=args.fraud_rate,
    )


if __name__ == "__main__":
    main()