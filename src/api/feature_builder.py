"""
Builds the 52 model features from a raw transaction request.
"""
import numpy as np
from datetime import datetime
from src.api.schemas import RawTransactionRequest


def build_features(tx: RawTransactionRequest) -> dict:
    """Convert a raw transaction into the 52 features the model expects."""

    ts = datetime.fromisoformat(tx.timestamp)
    hour = ts.hour
    day_of_week = ts.weekday()

    # --- Time features ---
    is_night = int(hour >= 22 or hour <= 5)
    is_early_morning = int(0 <= hour <= 5)
    is_business_hours = int(9 <= hour <= 17)
    is_weekend = int(day_of_week >= 5)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)

    # --- Amount features ---
    amount = tx.amount
    amount_log = np.log1p(amount)
    amount_very_small = int(amount < 100)
    amount_small = int(100 <= amount < 1000)
    amount_medium = int(1000 <= amount < 5000)
    amount_large = int(5000 <= amount < 20000)
    amount_very_large = int(amount >= 20000)

    # --- Velocity features ---
    tx_amount_1h_log = np.log1p(tx.tx_amount_1h)
    tx_amount_24h_log = np.log1p(tx.tx_amount_24h)
    avg_tx_amount_24h = (
        tx.tx_amount_24h / tx.tx_count_24h if tx.tx_count_24h > 0 else 0
    )

    # --- User history features ---
    user_total_amount = tx.user_avg_amount * tx.user_tx_number
    amount_deviation = (
        (amount - tx.user_avg_amount) / (tx.user_std_amount + 1)
        if tx.user_avg_amount > 0 else 0
    )
    amount_2x_avg = int(amount > 2 * tx.user_avg_amount)
    amount_3x_avg = int(amount > 3 * tx.user_avg_amount)

    # --- Receiver features ---
    is_new_receiver = (
        tx.is_new_receiver
        if tx.is_new_receiver is not None
        else int(tx.receiver_tx_count == 0)
    )

    # --- Device / location features ---
    device_changed_flag = int(tx.device_changed)
    location_changed_flag = int(tx.location_changed)
    device_or_location_changed = int(tx.device_changed or tx.location_changed)
    device_and_location_changed = int(tx.device_changed and tx.location_changed)

    # --- Interaction features ---
    high_amount_at_night = int(amount > tx.user_avg_amount * 2 and is_night)
    new_receiver_large_amount = int(is_new_receiver and amount > 5000)
    device_changed_unusual_amount = int(device_changed_flag and amount_2x_avg)
    high_velocity_large_amount = int(tx.tx_count_1h >= 3 and amount > 3000)
    night_device_change = int(is_night and device_changed_flag)

    return {
        # Time
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_night": is_night,
        "is_early_morning": is_early_morning,
        "is_business_hours": is_business_hours,
        "hour_sin": float(hour_sin),
        "hour_cos": float(hour_cos),
        "day_sin": float(day_sin),
        "day_cos": float(day_cos),
        # Amount
        "amount_raw": amount,
        "amount_log": float(amount_log),
        "amount_very_small": amount_very_small,
        "amount_small": amount_small,
        "amount_medium": amount_medium,
        "amount_large": amount_large,
        "amount_very_large": amount_very_large,
        # Velocity
        "tx_count_1h": tx.tx_count_1h,
        "tx_count_24h": tx.tx_count_24h,
        "tx_amount_1h": tx.tx_amount_1h,
        "tx_amount_24h": tx.tx_amount_24h,
        "tx_amount_1h_log": float(tx_amount_1h_log),
        "tx_amount_24h_log": float(tx_amount_24h_log),
        "time_since_last_tx": tx.time_since_last_tx,
        "avg_tx_amount_24h": avg_tx_amount_24h,
        # User history
        "user_tx_number": tx.user_tx_number,
        "user_total_amount": user_total_amount,
        "user_avg_amount": tx.user_avg_amount,
        "user_std_amount": tx.user_std_amount,
        "amount_deviation": float(amount_deviation),
        "amount_2x_avg": amount_2x_avg,
        "amount_3x_avg": amount_3x_avg,
        # Receiver
        "is_new_receiver": is_new_receiver,
        "receiver_tx_count": tx.receiver_tx_count,
        # Device / location
        "device_changed_flag": device_changed_flag,
        "location_changed_flag": location_changed_flag,
        "device_or_location_changed": device_or_location_changed,
        "device_and_location_changed": device_and_location_changed,
        # Interactions
        "high_amount_at_night": high_amount_at_night,
        "new_receiver_large_amount": new_receiver_large_amount,
        "device_changed_unusual_amount": device_changed_unusual_amount,
        "high_velocity_large_amount": high_velocity_large_amount,
        "night_device_change": night_device_change,
    }
