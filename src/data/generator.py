"""
Synthetic transaction data generator - FIXED VERSION

Generates realistic transaction data with both normal and fraudulent patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random
from faker import Faker

from src.data.patterns import get_random_fraud_pattern, FRAUD_PATTERNS

fake = Faker()


class TransactionGenerator:
    """Generate synthetic transaction data"""
    
    def __init__(
        self,
        num_users: int = 1000,
        num_transactions: int = 100000,
        fraud_rate: float = 0.02,  # 2% fraud rate (realistic)
        start_date: datetime = None,
        end_date: datetime = None,
        random_seed: int = 42
    ):
        """
        Args:
            num_users: Number of unique users to generate
            num_transactions: Total number of transactions
            fraud_rate: Percentage of fraudulent transactions (0.0 to 1.0)
            start_date: Start of transaction time range
            end_date: End of transaction time range
            random_seed: For reproducibility
        """
        self.num_users = num_users
        self.num_transactions = num_transactions
        self.fraud_rate = fraud_rate
        self.random_seed = random_seed
        
        # Default: 3 months of data
        self.end_date = end_date or datetime.now()
        self.start_date = start_date or (self.end_date - timedelta(days=90))
        
        # Set seeds for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        Faker.seed(random_seed)
        
        # Generate user profiles
        self.users = self._generate_users()
    
    def _generate_users(self) -> Dict[str, Dict[str, Any]]:
        """Generate user profiles with behavior patterns"""
        users = {}
        
        for i in range(self.num_users):
            user_id = f"user_{i:06d}"
            
            # Each user has typical behavior
            users[user_id] = {
                'user_id': user_id,
                'account_age_days': random.randint(30, 1000),
                
                # Transaction patterns (normal behavior)
                'avg_transaction_amount': np.random.lognormal(mean=6.5, sigma=0.8),  # ~600 KES
                'transaction_frequency': random.choice([2, 3, 4, 5, 7]),  # per week
                
                # Time preferences (when they usually transact)
                'active_hours': random.choice([
                    list(range(8, 18)),  # Business hours
                    list(range(10, 22)),  # Evening person
                    list(range(6, 15)),   # Morning person
                ]),
                
                # Common receivers (family, friends, merchants)
                'common_receivers': [
                    f"receiver_{random.randint(1000, 9999)}" 
                    for _ in range(random.randint(3, 8))
                ],
                
                # Location consistency
                'usual_location': fake.city(),
                'usual_device': f"device_{random.randint(100, 999)}",
            }
        
        return users
    
    def _generate_normal_transaction(
        self, 
        user_id: str, 
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Generate a normal (legitimate) transaction"""
        user = self.users[user_id]
        
        # Amount follows user's typical pattern with some variation
        amount = max(
            50,  # Minimum transaction
            np.random.normal(
                loc=user['avg_transaction_amount'],
                scale=user['avg_transaction_amount'] * 0.3
            )
        )
        
        # Usually to known receivers
        if random.random() < 0.8:  # 80% to known receivers
            receiver = random.choice(user['common_receivers'])
        else:  # 20% to new receivers (normal behavior)
            receiver = f"receiver_{random.randint(1000, 9999)}"
        
        # During usual hours
        hour = random.choice(user['active_hours'])
        tx_time = timestamp.replace(
            hour=hour,
            minute=random.randint(0, 59),
            second=random.randint(0, 59)
        )
        
        # FIXED: Legitimate users occasionally change devices/locations (5-10% of time)
        device_changed = random.random() < 0.07  # 7% chance (new phone, borrowed device)
        location_changed = random.random() < 0.08  # 8% chance (travel, moved)
        
        device_id = user['usual_device']
        location = user['usual_location']
        
        if device_changed:
            device_id = f"device_{random.randint(100, 999)}"
        
        if location_changed:
            location = fake.city()
        
        return {
            'transaction_id': fake.uuid4(),
            'user_id': user_id,
            'timestamp': tx_time,
            'amount': round(amount, 2),
            'receiver': receiver,
            'device_id': device_id,
            'location': location,
            'device_changed': device_changed,
            'location_changed': location_changed,
            'is_fraud': False,
            'fraud_pattern': None,
        }
    
    def _generate_fraud_transactions(
        self,
        user_id: str,
        timestamp: datetime
    ) -> List[Dict[str, Any]]:
        """Generate fraudulent transaction(s) for a user"""
        # Select fraud pattern
        pattern = get_random_fraud_pattern()
        
        # Generate fraud transactions based on pattern
        fraud_txs = pattern.generate(user_id, timestamp)
        
        # Add common fields
        for tx in fraud_txs:
            tx['transaction_id'] = fake.uuid4()
            tx['is_fraud'] = True
            tx['fraud_pattern'] = pattern.name
            
            # Set device/location if not specified
            if 'device_id' not in tx:
                user = self.users.get(tx['user_id'], {})
                if tx.get('device_changed', False):
                    tx['device_id'] = f"device_{random.randint(100, 999)}"
                else:
                    tx['device_id'] = user.get('usual_device', f"device_{random.randint(100, 999)}")
            
            if 'location' not in tx:
                user = self.users.get(tx['user_id'], {})
                if tx.get('location_changed', False):
                    tx['location'] = fake.city()
                else:
                    tx['location'] = user.get('usual_location', fake.city())
        
        return fraud_txs
    
    def generate(self) -> pd.DataFrame:
        """Generate complete transaction dataset"""
        print(f"Generating {self.num_transactions} transactions...")
        print(f"Fraud rate: {self.fraud_rate:.1%}")
        print(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        
        transactions = []
        
        # FIXED: Calculate correct number of transactions
        # We want final dataset to have fraud_rate % fraud
        # Since fraud patterns create multiple txs, we need to account for that
        
        # Average transactions per fraud event (varies by pattern)
        avg_txs_per_fraud_event = 5  # Rough estimate based on patterns
        
        # Calculate fraud events needed
        target_fraud_txs = int(self.num_transactions * self.fraud_rate)
        num_fraud_events = max(1, target_fraud_txs // avg_txs_per_fraud_event)
        
        # Remaining for normal transactions
        estimated_fraud_txs = num_fraud_events * avg_txs_per_fraud_event
        num_normal_transactions = self.num_transactions - estimated_fraud_txs
        
        # Generate normal transactions
        print(f"\nGenerating {num_normal_transactions:,} normal transactions...")
        for i in range(num_normal_transactions):
            # Random user
            user_id = random.choice(list(self.users.keys()))
            
            # Random timestamp in range
            timestamp = self.start_date + timedelta(
                seconds=random.randint(
                    0,
                    int((self.end_date - self.start_date).total_seconds())
                )
            )
            
            tx = self._generate_normal_transaction(user_id, timestamp)
            transactions.append(tx)
            
            if (i + 1) % 10000 == 0:
                print(f"  Generated {i + 1:,} normal transactions...")
        
        # Generate fraud transactions
        print(f"\nGenerating {num_fraud_events:,} fraud events...")
        for i in range(num_fraud_events):
            # Random user (could be any user - victim of attack)
            user_id = random.choice(list(self.users.keys()))
            
            # Random timestamp
            timestamp = self.start_date + timedelta(
                seconds=random.randint(
                    0,
                    int((self.end_date - self.start_date).total_seconds())
                )
            )
            
            fraud_txs = self._generate_fraud_transactions(user_id, timestamp)
            transactions.extend(fraud_txs)
            
            if (i + 1) % 500 == 0:
                print(f"  Generated {i + 1:,} fraud events...")
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add 'pattern' column for compatibility (same as fraud_pattern)
        df['pattern'] = df['fraud_pattern']
        
        # Print statistics
        print(f"\n{'='*60}")
        print("Dataset Statistics:")
        print(f"{'='*60}")
        print(f"Total transactions: {len(df):,}")
        print(f"Fraudulent: {df['is_fraud'].sum():,} ({df['is_fraud'].mean():.2%})")
        print(f"Legitimate: {(~df['is_fraud']).sum():,} ({(~df['is_fraud']).mean():.2%})")
        
        # Device/Location changes
        print(f"\nDevice Changes:")
        print(f"  Legitimate: {df[~df['is_fraud']]['device_changed'].sum():,} ({df[~df['is_fraud']]['device_changed'].mean():.1%})")
        print(f"  Fraudulent: {df[df['is_fraud']]['device_changed'].sum():,} ({df[df['is_fraud']]['device_changed'].mean():.1%})")
        
        print(f"\nLocation Changes:")
        print(f"  Legitimate: {df[~df['is_fraud']]['location_changed'].sum():,} ({df[~df['is_fraud']]['location_changed'].mean():.1%})")
        print(f"  Fraudulent: {df[df['is_fraud']]['location_changed'].sum():,} ({df[df['is_fraud']]['location_changed'].mean():.1%})")
        
        print(f"\nFraud patterns:")
        if df['is_fraud'].sum() > 0:
            print(df[df['is_fraud']]['fraud_pattern'].value_counts())
        
        print(f"\nAmount statistics:")
        print(f"  Legitimate: mean={df[~df['is_fraud']]['amount'].mean():.2f}, median={df[~df['is_fraud']]['amount'].median():.2f}")
        print(f"  Fraudulent: mean={df[df['is_fraud']]['amount'].mean():.2f}, median={df[df['is_fraud']]['amount'].median():.2f}")
        print(f"{'='*60}")
        
        return df


def generate_dataset(
    output_path: str = "data/raw/transactions.csv",
    num_users: int = 1000,
    num_transactions: int = 100000,
    fraud_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Convenience function to generate and save dataset.
    
    Args:
        output_path: Where to save the CSV
        num_users: Number of unique users
        num_transactions: Total transactions to generate
        fraud_rate: Percentage of fraudulent transactions
    
    Returns:
        Generated DataFrame
    """
    generator = TransactionGenerator(
        num_users=num_users,
        num_transactions=num_transactions,
        fraud_rate=fraud_rate
    )
    
    df = generator.generate()
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n✅ Dataset saved to: {output_path}")
    
    return df