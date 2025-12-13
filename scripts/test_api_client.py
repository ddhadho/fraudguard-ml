"""Test FraudGuard API with Python requests."""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000/api/v1"


def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/health")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ PASS")


def test_score_legitimate():
    """Test scoring legitimate transaction"""
    print("\n" + "="*60)
    print("TEST 2: Score Legitimate Transaction")
    print("="*60)
    
    payload = {
        "transaction_id": "TEST_LEGIT_001",
        "amount": 500,
        "hour": 14,
        "day_of_week": 2,
        "is_weekend": 0,
        "is_night": 0,
        "is_early_morning": 0,
        "is_business_hours": 1,
        "hour_sin": -0.5,
        "hour_cos": 0.87,
        "day_sin": 0.43,
        "day_cos": -0.9,
        "amount_log": 6.21,
        "amount_sqrt": 22.36,
        "amount_squared": 250000,
        "amount_deviation": 0,
        "amount_zscore": 0,
        "amount_percentile": 50,
        "is_round_amount": 1,
        "is_month_start": 0,
        "is_month_end": 0,
        "tx_count_1h": 1,
        "tx_count_6h": 3,
        "tx_count_24h": 5,
        "tx_count_7d": 25,
        "time_since_last_tx": 3600,
        "avg_tx_interval": 4000,
        "tx_velocity_1h": 1,
        "tx_velocity_24h": 5,
        "user_tx_count": 150,
        "user_avg_amount": 520,
        "user_std_amount": 200,
        "amount_deviation_from_user": -0.04,
        "amount_2x_avg": 0,
        "amount_3x_avg": 0,
        "user_fraud_rate": 0.01,
        "is_new_receiver": 0,
        "receiver_tx_count": 8,
        "device_changed_flag": 0,
        "device_change_count": 1,
        "location_changed_flag": 0,
        "location_change_count": 2,
        "high_amount_at_night": 0,
        "device_changed_unusual_amount": 0,
        "new_receiver_high_amount": 0,
        "velocity_spike": 0,
        "amount_velocity_interaction": 500
    }
    
    start_time = time.time()
    response = requests.post(f"{API_BASE_URL}/score", json=payload)
    latency = (time.time() - start_time) * 1000
    
    print(f"Status Code: {response.status_code}")
    print(f"Latency: {latency:.2f}ms")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    result = response.json()
    assert response.status_code == 200
    assert result["is_fraud"] == False, "Legitimate transaction flagged as fraud!"
    assert result["risk_level"] == "LOW"
    print("✅ PASS - Correctly identified as legitimate")


def test_score_fraud():
    """Test scoring fraudulent transaction"""
    print("\n" + "="*60)
    print("TEST 3: Score Fraudulent Transaction")
    print("="*60)
    
    payload = {
        "transaction_id": "TEST_FRAUD_001",
        "amount": 50000,
        "hour": 3,
        "day_of_week": 2,
        "is_weekend": 0,
        "is_night": 1,
        "is_early_morning": 1,
        "is_business_hours": 0,
        "hour_sin": 0.5,
        "hour_cos": 0.87,
        "day_sin": 0.43,
        "day_cos": -0.9,
        "amount_log": 10.82,
        "amount_sqrt": 223.61,
        "amount_squared": 2500000000,
        "amount_deviation": 5,
        "amount_zscore": 8,
        "amount_percentile": 95,
        "is_round_amount": 1,
        "is_month_start": 0,
        "is_month_end": 0,
        "tx_count_1h": 8,
        "tx_count_6h": 15,
        "tx_count_24h": 20,
        "tx_count_7d": 25,
        "time_since_last_tx": 120,
        "avg_tx_interval": 180,
        "tx_velocity_1h": 8,
        "tx_velocity_24h": 20,
        "user_tx_count": 150,
        "user_avg_amount": 520,
        "user_std_amount": 200,
        "amount_deviation_from_user": 95,
        "amount_2x_avg": 1,
        "amount_3x_avg": 1,
        "user_fraud_rate": 0.01,
        "is_new_receiver": 1,
        "receiver_tx_count": 0,
        "device_changed_flag": 1,
        "device_change_count": 5,
        "location_changed_flag": 1,
        "location_change_count": 5,
        "high_amount_at_night": 1,
        "device_changed_unusual_amount": 1,
        "new_receiver_high_amount": 1,
        "velocity_spike": 1,
        "amount_velocity_interaction": 50000
    }
    
    start_time = time.time()
    response = requests.post(f"{API_BASE_URL}/score", json=payload)
    latency = (time.time() - start_time) * 1000
    
    print(f"Status Code: {response.status_code}")
    print(f"Latency: {latency:.2f}ms")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    result = response.json()
    assert response.status_code == 200
    assert result["is_fraud"] == True, "Fraud not detected!"
    assert result["risk_level"] in ["HIGH", "MEDIUM"]
    print("✅ PASS - Correctly identified as fraud")


def test_score_detailed():
    """Test detailed scoring with explainability"""
    print("\n" + "="*60)
    print("TEST 4: Detailed Scoring with Explainability")
    print("="*60)
    
    payload = {
        "transaction_id": "TEST_FRAUD_002",
        "amount": 50000,
        "hour": 3,
        "day_of_week": 2,
        "is_weekend": 0,
        "is_night": 1,
        "is_early_morning": 1,
        "is_business_hours": 0,
        "hour_sin": 0.5,
        "hour_cos": 0.87,
        "day_sin": 0.43,
        "day_cos": -0.9,
        "amount_log": 10.82,
        "amount_sqrt": 223.61,
        "amount_squared": 2500000000,
        "amount_deviation": 5,
        "amount_zscore": 8,
        "amount_percentile": 95,
        "is_round_amount": 1,
        "is_month_start": 0,
        "is_month_end": 0,
        "tx_count_1h": 8,
        "tx_count_6h": 15,
        "tx_count_24h": 20,
        "tx_count_7d": 25,
        "time_since_last_tx": 120,
        "avg_tx_interval": 180,
        "tx_velocity_1h": 8,
        "tx_velocity_24h": 20,
        "user_tx_count": 150,
        "user_avg_amount": 520,
        "user_std_amount": 200,
        "amount_deviation_from_user": 95,
        "amount_2x_avg": 1,
        "amount_3x_avg": 1,
        "user_fraud_rate": 0.01,
        "is_new_receiver": 1,
        "receiver_tx_count": 0,
        "device_changed_flag": 1,
        "device_change_count": 5,
        "location_changed_flag": 1,
        "location_change_count": 5,
        "high_amount_at_night": 1,
        "device_changed_unusual_amount": 1,
        "new_receiver_high_amount": 1,
        "velocity_spike": 1,
        "amount_velocity_interaction": 50000
    }
    
    response = requests.post(f"{API_BASE_URL}/score/detailed", json=payload)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    result = response.json()
    assert response.status_code == 200
    assert "reason_codes" in result
    assert len(result["reason_codes"]) > 0
    print(f"✅ PASS - Found {len(result['reason_codes'])} fraud indicators")


def test_batch_scoring():
    """Test batch scoring"""
    print("\n" + "="*60)
    print("TEST 5: Batch Scoring")
    print("="*60)
    
    # Create 10 test transactions
    transactions = []
    for i in range(10):
        tx = {
            "transaction_id": f"BATCH_TEST_{i:03d}",
            "amount": 500 + i * 100,
            "hour": 14,
            "day_of_week": 2,
            "is_weekend": 0,
            "is_night": 0,
            "is_early_morning": 0,
            "is_business_hours": 1,
            "hour_sin": -0.5,
            "hour_cos": 0.87,
            "day_sin": 0.43,
            "day_cos": -0.9,
            "amount_log": 6.21,
            "amount_sqrt": 22.36,
            "amount_squared": 250000,
            "amount_deviation": 0,
            "amount_zscore": 0,
            "amount_percentile": 50,
            "is_round_amount": 1,
            "is_month_start": 0,
            "is_month_end": 0,
            "tx_count_1h": 1,
            "tx_count_6h": 3,
            "tx_count_24h": 5,
            "tx_count_7d": 25,
            "time_since_last_tx": 3600,
            "avg_tx_interval": 4000,
            "tx_velocity_1h": 1,
            "tx_velocity_24h": 5,
            "user_tx_count": 150,
            "user_avg_amount": 520,
            "user_std_amount": 200,
            "amount_deviation_from_user": -0.04,
            "amount_2x_avg": 0,
            "amount_3x_avg": 0,
            "user_fraud_rate": 0.01,
            "is_new_receiver": 0,
            "receiver_tx_count": 8,
            "device_changed_flag": 0,
            "device_change_count": 1,
            "location_changed_flag": 0,
            "location_change_count": 2,
            "high_amount_at_night": 0,
            "device_changed_unusual_amount": 0,
            "new_receiver_high_amount": 0,
            "velocity_spike": 0,
            "amount_velocity_interaction": 500
        }
        transactions.append(tx)
    
    payload = {
        "transactions": transactions,
        "include_reasons": False
    }
    
    start_time = time.time()
    response = requests.post(f"{API_BASE_URL}/score/batch", json=payload)
    latency = (time.time() - start_time) * 1000
    
    print(f"Status Code: {response.status_code}")
    print(f"Total Latency: {latency:.2f}ms")
    
    result = response.json()
    print(f"Transactions Processed: {result['total_transactions']}")
    print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
    print(f"Avg per transaction: {result['processing_time_ms']/10:.2f}ms")
    
    assert response.status_code == 200
    assert result["total_transactions"] == 10
    assert len(result["results"]) == 10
    print("✅ PASS - Batch scoring successful")


def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*60)
    print("TEST 6: Model Info")
    print("="*60)
    
    response = requests.get(f"{API_BASE_URL}/model/info")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✅ PASS")


def test_error_handling():
    """Test error handling"""
    print("\n" + "="*60)
    print("TEST 7: Error Handling")
    print("="*60)
    
    # Test with invalid data (negative amount)
    payload = {
        "amount": -100,  # Invalid!
        "hour": 14
        # Missing required fields
    }
    
    response = requests.post(f"{API_BASE_URL}/score", json=payload)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 422  # Validation error
    print("✅ PASS - Error handling works correctly")


def main():
    """Run all tests"""
    print("="*60)
    print("🚀 FraudGuard API Test Suite")
    print("="*60)
    
    try:
        test_health()
        test_score_legitimate()
        test_score_fraud()
        test_score_detailed()
        test_batch_scoring()
        test_model_info()
        test_error_handling()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\n🎉 Your API is working perfectly!")
        print("\nNext steps:")
        print("  1. Check out the docs: http://localhost:8000/docs")
        print("  2. Test in browser")
        print("  3. Ready for Docker & deployment!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("   Make sure the API is running:")
        print("   uvicorn src.api.main:app --reload")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()