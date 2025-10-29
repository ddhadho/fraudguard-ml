# Feature Engineering Documentation

## Overview

This document describes the 43 features engineered for fraud detection in mobile money transactions.

## Feature Categories

### 1. Time Features (10 features)

Features capturing temporal patterns of transactions.

| Feature | Description | Type | Fraud Signal |
|---------|-------------|------|--------------|
| `hour` | Hour of day (0-23) | Numeric | Fraud peaks at night (hour 7 avg vs 12 for legit) |
| `day_of_week` | Day of week (0=Mon, 6=Sun) | Numeric | Weekly patterns |
| `is_weekend` | Weekend indicator | Binary | Fraud patterns may differ |
| `is_night` | Night time (22:00-05:00) | Binary | **Strong signal**: 52% fraud at night vs 0% legit |
| `is_early_morning` | Early morning (00:00-05:00) | Binary | Peak fraud time |
| `is_business_hours` | Business hours (09:00-17:00) | Binary | Lower fraud during business hours |
| `hour_sin` | Cyclical hour encoding (sin) | Numeric | Captures hour cyclicity |
| `hour_cos` | Cyclical hour encoding (cos) | Numeric | Captures hour cyclicity |
| `day_sin` | Cyclical day encoding (sin) | Numeric | Captures weekly cyclicity |
| `day_cos` | Cyclical day encoding (cos) | Numeric | Captures weekly cyclicity |

**Key Insight:** Fraud transactions occur predominantly at night (52% vs 0% for legitimate).

---

### 2. Amount Features (7 features)

Features describing transaction amounts and their characteristics.

| Feature | Description | Type | Fraud Signal |
|---------|-------------|------|--------------|
| `amount_raw` | Raw transaction amount | Numeric | Fraud avg: 13,547 KES vs 931 KES (14.5x higher) |
| `amount_log` | Log-transformed amount | Numeric | Handles skewed distribution |
| `amount_very_small` | Amount < 100 KES | Binary | Rare in fraud |
| `amount_small` | 100 ≤ amount < 1,000 KES | Binary | Common in legitimate |
| `amount_medium` | 1,000 ≤ amount < 5,000 KES | Binary | Mixed |
| `amount_large` | 5,000 ≤ amount < 20,000 KES | Binary | Higher in fraud |
| `amount_very_large` | Amount ≥ 20,000 KES | Binary | **Strong fraud signal** |

**Key Insight:** Fraudulent transactions are 14.5x higher in amount on average.

---

### 3. Velocity Features (8 features)

Features measuring transaction frequency and patterns over time windows.

| Feature | Description | Type | Fraud Signal |
|---------|-------------|------|--------------|
| `tx_count_1h` | Transactions in last 1 hour | Numeric | **Strong**: Fraud avg 3.0 vs 0.1 (30x higher) |
| `tx_count_24h` | Transactions in last 24 hours | Numeric | Fraud avg 3.9 vs 1.1 (3.5x higher) |
| `tx_amount_1h` | Total amount in last 1 hour | Numeric | High velocity amount = fraud |
| `tx_amount_24h` | Total amount in last 24 hours | Numeric | Cumulative amount signal |
| `tx_amount_1h_log` | Log of tx_amount_1h | Numeric | **2nd most important** in RF |
| `tx_amount_24h_log` | Log of tx_amount_24h | Numeric | Normalized velocity |
| `time_since_last_tx` | Minutes since last transaction | Numeric | Fraud: 496 min vs 1292 min (rapid-fire) |
| `avg_tx_amount_24h` | Average amount in 24h window | Numeric | Context for current transaction |

**Key Insight:** Velocity is the strongest fraud signal - fraudsters make rapid successive transactions.

---

### 4. User History Features (7 features)

Features comparing current transaction to user's historical behavior.

| Feature | Description | Type | Fraud Signal |
|---------|-------------|------|--------------|
| `user_tx_number` | User's Nth transaction | Numeric | Fraud often on early transactions |
| `user_total_amount` | Cumulative total amount sent | Numeric | User activity level |
| `user_avg_amount` | User's average transaction amount | Numeric | Baseline for comparison |
| `user_std_amount` | Std dev of user's amounts | Numeric | User's consistency |
| `amount_deviation` | Z-score of current amount | Numeric | **Fraud avg: 2.17 vs -0.04** |
| `amount_2x_avg` | Amount > 2x user average | Binary | **Strongest correlation (0.74)** |
| `amount_3x_avg` | Amount > 3x user average | Binary | **3rd strongest (0.67)** |

**Key Insight:** Transactions deviating from user's typical behavior are highly indicative of fraud.

---

### 5. Receiver Features (2 features)

Features about the transaction recipient.

| Feature | Description | Type | Fraud Signal |
|---------|-------------|------|--------------|
| `is_new_receiver` | Receiver never seen before | Binary | **99.9% of fraud** to new receivers |
| `receiver_tx_count` | Times sent to this receiver | Numeric | Low count = suspicious |

**Key Insight:** Almost all fraud involves sending to new, unknown receivers.

---

### 6. Device/Location Features (4 features)

Features tracking device and location changes.

| Feature | Description | Type | Fraud Signal |
|---------|-------------|------|--------------|
| `device_changed_flag` | Device ID changed | Binary | 32% fraud vs 7% legit |
| `location_changed_flag` | Location changed | Binary | 32% fraud vs 8% legit |
| `device_or_location_changed` | Either changed | Binary | Combined signal |
| `device_and_location_changed` | Both changed | Binary | **Strong fraud indicator** |

**Key Insight:** Device/location changes are more common in fraud but not deterministic (32% vs 7%).

---

### 7. Interaction Features (5 features)

Combinations of other features that capture fraud patterns.

| Feature | Description | Fraud Rate When True | Lift |
|---------|-------------|---------------------|------|
| `high_amount_at_night` | High amount + night time | **100%** | 40x |
| `new_receiver_large_amount` | New receiver + large amount | 84% | 34x |
| `device_changed_unusual_amount` | Device change + unusual amount | **99.5%** | 40x |
| `high_velocity_large_amount` | High velocity + large amount | **100%** | 40x |
| `night_device_change` | Night + device change | **100%** | 40x |

**⚠️ Note:** These features show near-perfect fraud rates due to deterministic patterns in synthetic data. Real-world production would show 60-80% precision, not 99%+.

---

## Feature Importance Rankings

### By Correlation with Fraud Target

| Rank | Feature | Correlation |
|------|---------|-------------|
| 1 | `amount_2x_avg` | 0.739 |
| 2 | `is_night` | 0.716 |
| 3 | `amount_3x_avg` | 0.668 |
| 4 | `is_early_morning` | 0.653 |
| 5 | `new_receiver_large_amount` | 0.617 |
| 6 | `high_amount_at_night` | 0.596 |
| 7 | `tx_count_1h` | 0.587 |
| 8 | `amount_raw` | 0.571 |
| 9 | `tx_amount_1h` | 0.547 |
| 10 | `night_device_change` | 0.540 |

### By Random Forest Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `tx_amount_1h` | 0.114 |
| 2 | `tx_amount_1h_log` | 0.109 |
| 3 | `amount_2x_avg` | 0.101 |
| 4 | `is_night` | 0.077 |
| 5 | `time_since_last_tx` | 0.061 |
| 6 | `is_early_morning` | 0.051 |
| 7 | `amount_3x_avg` | 0.049 |
| 8 | `tx_count_1h` | 0.048 |
| 9 | `amount_raw` | 0.046 |
| 10 | `hour` | 0.044 |

---

## Key Findings

### 1. Velocity is King
- Transaction velocity (frequency in short time windows) is the strongest signal
- `tx_amount_1h` and `tx_count_1h` consistently rank in top features
- Fraudulent transactions occur 30x more frequently in 1-hour windows

### 2. Amount Anomalies Matter
- Transactions deviating from user's baseline are highly suspicious
- `amount_2x_avg` has 0.74 correlation with fraud
- Fraud amounts are 14.5x higher on average

### 3. Temporal Patterns
- Night transactions (22:00-05:00) are strong fraud indicators
- 52% of fraud occurs at night vs 0% of legitimate transactions
- Early morning (00:00-05:00) is peak fraud time

### 4. Behavioral Signals
- 99.9% of fraud involves new receivers
- Device/location changes occur 4-5x more in fraud
- Rapid-fire transactions (short time since last) indicate fraud

### 5. Interaction Effects
- Combinations amplify signals: night + high amount = 100% fraud rate
- Device change + unusual amount = 99.5% fraud rate
- Multiple weak signals together create strong detection

---

## Feature Selection Recommendations

### Tier 1: Essential (Must Include)
- `tx_amount_1h`, `tx_count_1h` - Velocity signals
- `amount_2x_avg`, `amount_3x_avg` - Deviation signals
- `is_night`, `is_early_morning` - Time signals
- `amount_raw`, `amount_log` - Amount features
- `is_new_receiver` - Receiver signal

### Tier 2: Important
- `time_since_last_tx` - Velocity context
- `amount_deviation` - Statistical deviation
- `device_changed_flag`, `location_changed_flag` - Context changes
- `tx_count_24h`, `tx_amount_24h` - Longer velocity windows
- Interaction features (if not overfitting)

### Tier 3: Optional
- Cyclical encodings (if using tree-based models, may not help)
- User history aggregates
- Additional time buckets

### Consider Dropping
- Features with very low importance (<0.01)
- Redundant features (e.g., keep `amount_log` OR `amount_raw`, not both)
- Interaction features that cause overfitting

---

## Production Considerations

### Real-Time Feature Computation

**Fast Features (<1ms):**
- Time features (computed from timestamp)
- Amount features (from transaction itself)
- Device/location flags (from transaction)

**Medium Speed (1-10ms):**
- User history features (requires user profile lookup from cache)
- Velocity features (requires recent transaction history)

**Slow Features (>10ms):**
- Receiver history (requires database queries)
- Graph features (requires network analysis - Phase 2)

### Feature Store Requirements

For production deployment, need to maintain:
```
User Profile Store (Redis/DynamoDB):
- user_avg_amount
- user_std_amount
- user_tx_number
- usual_device
- usual_location

Recent Transaction Cache (Redis):
- Last N transactions for velocity calculations
- Sliding windows (1h, 24h)
```

### Feature Drift Monitoring

Monitor these features for distribution shift:
- `amount_raw` distribution
- `tx_count_1h` distribution
- `is_night` percentage
- `is_new_receiver` percentage

If distributions change significantly (>10%), retrain model.

---

## Limitations & Future Improvements

### Current Limitations

1. **Synthetic Data Perfection**: Interaction features show 99-100% fraud rates due to deterministic pattern generation. Real-world would be 60-80%.

2. **No Network Features**: Missing graph-based features (transaction networks, mule account detection) - planned for Phase 2.

3. **Static User Profiles**: User behavior may change over time (not captured).

4. **Limited Context**: No merchant category, device fingerprint, geolocation coordinates.

### Future Enhancements

**Phase 2 Features:**
- Graph neural network features
- Network centrality metrics
- Community detection scores
- Money flow analysis

**Additional Features:**
- Device fingerprinting (browser, OS, screen resolution)
- Geolocation coordinates (not just city names)
- Merchant category codes
- Time since account creation
- Seasonal patterns (month-end salary effects)

**Advanced Techniques:**
- Embeddings for categorical features (device_id, location)
- Recurrent features (LSTM over transaction sequences)
- Anomaly scores from isolation forests
- Clustering-based features

---

## References & Resources

- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Practical Lessons from Predicting Clicks on Ads at Facebook](https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/)
- [Real-time Data Infrastructure at Uber](https://eng.uber.com/real-time-data-infrastructure/)