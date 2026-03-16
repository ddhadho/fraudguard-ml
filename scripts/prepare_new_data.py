import pandas as pd
from pathlib import Path

# Load your CSV
df = pd.read_csv("/home/dhadho/Downloads/fraud_detection_dataset.csv")

# Rename columns to match what the feature engineer expects
df = df.rename(columns={
    'transaction_date': 'timestamp',
    'merchant': 'receiver',
    'device_type': 'device_id',
})

# Create missing columns
# device_changed: flag if device_type changed from previous transaction for same user
df = df.sort_values(['user_id', 'timestamp'])
df['prev_device'] = df.groupby('user_id')['device_id'].shift(1)
df['device_changed'] = (df['device_id'] != df['prev_device']).fillna(False)

# location_changed: use country or ip_address as proxy
df['prev_country'] = df.groupby('user_id')['country'].shift(1)
df['location_changed'] = (df['country'] != df['prev_country']).fillna(False)

# Drop columns the script doesn't need
df = df.drop(columns=['prev_device', 'prev_country', 
                       'transaction_time', 'card_type', 'card_last_4',
                       'ip_address', 'merchant_category', 'user_age',
                       'account_age_days', 'transaction_count_24h',
                       'avg_transaction_amount', 'distance_from_last_transaction',
                       'merchant_reputation_score', 'is_weekend',
                       'is_international', 'country'], errors='ignore')

# Save to where the pipeline expects it
Path("data/raw").mkdir(parents=True, exist_ok=True)
df.to_csv("data/raw/transactions.csv", index=False)

print(f"✅ Done. {len(df):,} transactions saved to data/raw/transactions.csv")
print(f"Columns: {list(df.columns)}")
