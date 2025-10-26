"""
Fraud pattern definitions for synthetic data generation.

Each pattern represents a real-world fraud scenario with specific characteristics.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime, timedelta
import random

@dataclass
class FraudPattern:
    """Base class for fraud patterns"""
    name: str
    description: str
    probability: float  # How common is this pattern
    
    def generate(self, user_id: str, timestamp: datetime) -> List[Dict[str, Any]]:
        """Generate fraudulent transactions for this pattern"""
        raise NotImplementedError


class SIMSwapPattern(FraudPattern):
    """
    SIM Swap Attack:
    - Attacker obtains victim's SIM card
    - Changes device suddenly
    - Makes large unusual transactions
    - Often at unusual hours (late night)
    - Sends to new/unknown receivers
    """
    
    def __init__(self):
        super().__init__(
            name="sim_swap",
            description="Attacker swaps SIM card and drains account",
            probability=0.35
        )
    
    def generate(self, user_id: str, timestamp: datetime) -> List[Dict[str, Any]]:
        """Generate SIM swap fraud transactions"""
        transactions = []
        
        # Attack happens late night/early morning
        attack_hour = random.choice([0, 1, 2, 3, 4, 5, 23])
        attack_time = timestamp.replace(hour=attack_hour, minute=random.randint(0, 59))
        
        # Multiple rapid transactions (drain account quickly)
        num_transactions = random.randint(3, 7)
        
        for i in range(num_transactions):
            # Each transaction a few minutes apart
            tx_time = attack_time + timedelta(minutes=i * random.randint(2, 8))
            
            # Large amounts (trying to drain account)
            amount = random.uniform(5000, 50000)
            
            transactions.append({
                'user_id': user_id,
                'timestamp': tx_time,
                'amount': amount,
                'receiver': f'mule_account_{random.randint(1000, 9999)}',
                'device_changed': True,  # Key indicator
                'location_changed': True,
                'pattern': self.name
            })
        
        return transactions


class VelocityAttackPattern(FraudPattern):
    """
    Velocity Attack:
    - Rapid succession of transactions
    - Much higher frequency than normal
    - Often to multiple different receivers
    - Trying to move money before detection
    """
    
    def __init__(self):
        super().__init__(
            name="velocity_attack",
            description="Rapid-fire transactions before detection",
            probability=0.25
        )
    
    def generate(self, user_id: str, timestamp: datetime) -> List[Dict[str, Any]]:
        """Generate velocity attack transactions"""
        transactions = []
        
        # Many transactions in short time window
        num_transactions = random.randint(8, 15)
        
        for i in range(num_transactions):
            # Very close together (minutes apart)
            tx_time = timestamp + timedelta(minutes=i * random.randint(1, 3))
            
            # Medium amounts (not too suspicious individually)
            amount = random.uniform(1000, 5000)
            
            # Different receivers (spreading money)
            receiver = f'receiver_{random.randint(1000, 9999)}'
            
            transactions.append({
                'user_id': user_id,
                'timestamp': tx_time,
                'amount': amount,
                'receiver': receiver,
                'device_changed': False,
                'location_changed': False,
                'pattern': self.name
            })
        
        return transactions


class UnusualAmountPattern(FraudPattern):
    """
    Unusual Amount:
    - Single large transaction
    - Much higher than user's normal behavior
    - Often to new receiver
    - May be account takeover or stolen credentials
    """
    
    def __init__(self):
        super().__init__(
            name="unusual_amount",
            description="Transaction amount much higher than normal",
            probability=0.20
        )
    
    def generate(self, user_id: str, timestamp: datetime) -> List[Dict[str, Any]]:
        """Generate unusual amount fraud transaction"""
        # Single large transaction
        amount = random.uniform(20000, 100000)
        
        return [{
            'user_id': user_id,
            'timestamp': timestamp,
            'amount': amount,
            'receiver': f'new_receiver_{random.randint(1000, 9999)}',
            'device_changed': random.choice([True, False]),
            'location_changed': random.choice([True, False]),
            'pattern': self.name
        }]


class MuleAccountPattern(FraudPattern):
    """
    Mule Account:
    - Account receives money from many different senders
    - Then quickly transfers out (cash-out)
    - Classic money laundering pattern
    - Network-based fraud (needs graph to detect well)
    """
    
    def __init__(self):
        super().__init__(
            name="mule_account",
            description="Account used for money laundering",
            probability=0.15
        )
    
    def generate(self, user_id: str, timestamp: datetime) -> List[Dict[str, Any]]:
        """Generate mule account transactions"""
        transactions = []
        
        # This user is the MULE - receives from many, sends to few
        # For synthetic data, we'll mark these as fraud
        # In real system, graph analysis would detect this pattern
        
        # Multiple incoming transactions (different senders)
        num_incoming = random.randint(5, 12)
        for i in range(num_incoming):
            tx_time = timestamp + timedelta(hours=i * random.randint(1, 4))
            amount = random.uniform(2000, 8000)
            
            transactions.append({
                'user_id': f'sender_{random.randint(1000, 9999)}',  # Different senders
                'timestamp': tx_time,
                'amount': amount,
                'receiver': user_id,  # All to this mule account
                'device_changed': False,
                'location_changed': False,
                'pattern': f'{self.name}_incoming'
            })
        
        # Quick cash-out to final destination
        cash_out_time = timestamp + timedelta(hours=num_incoming * 4)
        total_amount = sum(tx['amount'] for tx in transactions) * 0.9  # Keep 10% fee
        
        transactions.append({
            'user_id': user_id,
            'timestamp': cash_out_time,
            'amount': total_amount,
            'receiver': f'final_destination_{random.randint(100, 999)}',
            'device_changed': False,
            'location_changed': False,
            'pattern': f'{self.name}_cashout'
        })
        
        return transactions


class NightTransactionPattern(FraudPattern):
    """
    Night Transaction:
    - Transaction at unusual hour (2am-5am)
    - When user normally doesn't transact
    - Could be legitimate, but suspicious
    """
    
    def __init__(self):
        super().__init__(
            name="night_transaction",
            description="Transaction at unusual hour",
            probability=0.05
        )
    
    def generate(self, user_id: str, timestamp: datetime) -> List[Dict[str, Any]]:
        """Generate night transaction"""
        # Late night/early morning
        night_hour = random.choice([2, 3, 4])
        tx_time = timestamp.replace(hour=night_hour, minute=random.randint(0, 59))
        
        amount = random.uniform(3000, 15000)
        
        return [{
            'user_id': user_id,
            'timestamp': tx_time,
            'amount': amount,
            'receiver': f'receiver_{random.randint(1000, 9999)}',
            'device_changed': random.choice([True, False]),
            'location_changed': False,
            'pattern': self.name
        }]


# Registry of all fraud patterns
FRAUD_PATTERNS = [
    SIMSwapPattern(),
    VelocityAttackPattern(),
    UnusualAmountPattern(),
    MuleAccountPattern(),
    NightTransactionPattern(),
]


def get_random_fraud_pattern() -> FraudPattern:
    """Select a random fraud pattern based on probabilities"""
    patterns = FRAUD_PATTERNS
    probabilities = [p.probability for p in patterns]
    
    # Normalize probabilities
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]
    
    return random.choices(patterns, weights=probabilities, k=1)[0]