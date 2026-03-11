# 🛡️ FraudGuard

**Real-time fraud detection for mobile money transactions using Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

A production-ready ML system that detects fraudulent transactions in mobile money platforms (like M-Pesa) with **99.8% detection rate** and only **0.06% false alarms**.

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Precision** | 97.6% |
| **Recall** | 99.8% |
| **F1-Score** | 98.7% |
| **False Alarm Rate** | 0.06% |
| **Inference Time** | <5ms per transaction |
| **Throughput** | 100K+ transactions in 0.4 seconds |

### Business Impact

**Out of 100,000 daily transactions:**
- Fraud detected: **2,511 / 2,516** (99.8%)
- False alarms: **62 / 98,000** (0.06%)
- Estimated savings: **21.25M KES** (based on 8,500 KES avg fraud)

---

## Key Features

-  **Real-time detection** - Sub-5ms inference time
- **High accuracy** - 97.6% precision, 99.8% recall
- **Explainable AI** - Fraud indicators and feature importance
- **Production API** - FastAPI with auto-generated docs
- **Docker ready** - Containerized and cloud-deployable
- **Scalable** - Optimized batch processing

---

## Architecture
```
┌──────────────────────────────────────────────────────┐
│              FraudGuard ML Pipeline                  │
├──────────────────────────────────────────────────────┤
│                                                       │
│  1. Data Generation                                  │
│     • Synthetic transactions (100K+)                 │
│     • 5 realistic fraud patterns                     │
│     • SIM swap, velocity attacks, etc.               │
│                                                       │
│  2. Feature Engineering (52 features)                │
│     • Time features (hour, day, cyclical)            │
│     • Amount features (raw, log, bins)               │
│     • Velocity (transaction frequency)               │
│     • User history (deviation from norm)             │
│     • Interaction features (combined signals)        │
│                                                       │
│  3. ML Model (XGBoost)                               │
│     • 200 trees, class weighting                     │
│     • 5-fold cross-validation                        │
│     • Optimized hyperparameters                      │
│                                                       │
│  4. FastAPI Service                                  │
│     • REST endpoints                                 │
│     • Auto-generated docs                            │
│     • Request validation                             │
│                                                       │
└──────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)

### Option 1: Local Installation
```bash
# Clone repository
git clone https://github.com/ddhadho/fraudguard-ml.git
cd fraudguard-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate data and train model
python scripts/generate_data.py
python scripts/engineer_features.py
python scripts/train_model.py

# Start API
uvicorn src.api.main:app --reload
```

**API available at:** http://localhost:8000  
**Docs:** http://localhost:8000/docs

### Option 2: Docker (Recommended)
```bash
# Clone repository
git clone https://github.com/ddhadho/fraudguard-ml.git
cd fraudguard-ml

# Start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## API Usage

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

### Score a Transaction
```bash
curl -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_001",
    "amount": 5000,
    "hour": 14,
    "is_new_receiver": 1,
    "device_changed_flag": 0,
    ... (43 features total)
  }'
```

**Response:**
```json
{
  "transaction_id": "TXN_001",
  "is_fraud": true,
  "fraud_score": 0.87,
  "risk_level": "HIGH",
  "threshold": 0.75,
  "processing_time_ms": 3.2
}
```

### Detailed Scoring with Explainability
```bash
curl -X POST http://localhost:8000/api/v1/score/detailed \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

**Returns fraud indicators:**
```json
{
  "is_fraud": true,
  "fraud_score": 0.87,
  "reason_codes": [
    "amount_much_higher_than_usual",
    "new_receiver_high_amount",
    "device_change_detected"
  ],
  "top_features": [
    {"feature": "amount_deviation", "value": 8.62}
  ]
}
```

---

## Fraud Patterns Detected

| Pattern | Description | Detection Rate |
|---------|-------------|----------------|
| **SIM Swap** | Attacker obtains victim's SIM | 95%+ |
| **Velocity Attack** | Rapid transactions before detection | 98%+ |
| **Account Takeover** | Compromised credentials | 92%+ |
| **Mule Networks** | Money laundering chains | 85%+ |
| **Night Transactions** | Large amounts at unusual hours | 97%+ |

---

## Project Structure
```
fraudguard-ml/
├── src/
│   ├── api/              # FastAPI application
│   ├── data/             # Data generation
│   ├── features/         # Feature engineering
│   └── models/           # ML models
├── scripts/              # Training & testing
├── notebooks/            # Jupyter analysis
├── models/               # Trained models
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Tech Stack

- **ML Framework:** XGBoost 2.0
- **API:** FastAPI 0.104
- **Validation:** Pydantic
- **Server:** Uvicorn
- **Data:** Pandas, NumPy
- **Containerization:** Docker

---

## Testing
```bash
# Test predictor
python scripts/test_predictor.py

# Test API
python scripts/test_api_client.py
```

---

## Deployment

### Docker
```bash
# Build image
docker build -t fraudguard-api .

# Run container
docker run -d -p 8000:8000 fraudguard-api
```

---

## Model Details

### Features (52 total)

- **Time** (10): Hour, day, weekend, night, cyclical encoding
- **Amount** (7): Raw, log-transformed, bins
- **Velocity** (8): Transaction counts in time windows
- **User History** (7): Cumulative stats, deviation
- **Device/Location** (4): Change detection
- **Interactions** (5): Combined fraud signals

### Training Details

- **Algorithm:** XGBoost Gradient Boosting
- **Trees:** 200
- **Balancing:** scale_pos_weight (45.09)
- **Validation:** 5-fold stratified CV
- **Dataset:** 100K+ transactions, 2.5% fraud rate

### Top Features

1. `amount_deviation` (14.2%) - Deviation from user norm
2. `tx_count_1h` (9.8%) - Transaction velocity
3. `device_changed_flag` (8.7%) - Device change
4. `is_night` (7.6%) - Night-time flag
5. `is_new_receiver` (6.8%) - New receiver

---

## Roadmap

### Phase 2: Graph Neural Networks
- [ ] Transaction network graph construction
- [ ] GraphSAGE/GAT implementation
- [ ] Fraud ring detection
- [ ] Network-based scoring

### Phase 3: Production Enhancements
- [ ] Real-time feedback loop
- [ ] A/B testing framework
- [ ] Model monitoring & drift detection
- [ ] Multi-model ensemble

---

## License

MIT License - see [LICENSE](LICENSE) file

---
