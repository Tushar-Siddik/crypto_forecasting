# Cryptocurrency Forecasting System - API Reference

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Endpoints](#endpoints)
   - [Health Check](#health-check)
   - [Models](#models)
   - [Predictions](#predictions)
   - [Model Management](#model-management)
4. [Request/Response Formats](#requestresponse-formats)
5. [Error Codes](#error-codes)
6. [Rate Limiting](#rate-limiting)
7. [Examples](#examples)

## Overview

The Cryptocurrency Forecasting System provides a RESTful API for real-time cryptocurrency price prediction. The API supports multiple model architectures and provides both point predictions and confidence intervals.

### Base URL

```
http://localhost:8000
```

### Content-Type

All requests must use the `application/json` content type.

## Authentication

Currently, the API does not require authentication for basic usage. For production deployments, you can enable API key authentication:

```python
# Add API key to request headers
headers = {
    "X-API-Key": "your-api-key-here"
}
```

## Endpoints

### Health Check

Check if the API is running and get system status.

#### Endpoint
```
GET /health
```

#### Response
```json
{
  "status": "healthy",
  "models_loaded": ["lstm_attention_BTC-USD", "transformer_ETH-USD"],
  "last_updated": "2023-12-01T12:00:00Z"
}
```

#### Status Codes
- `200 OK`: API is healthy
- `503 Service Unavailable`: API is down

---

### Models

Get information about available models and their status.

#### List All Models
```
GET /models
```

##### Response
```json
[
  "lstm_attention_BTC-USD",
  "transformer_ETH-USD",
  "gru_attention_BNB-USD",
  "ensemble_SOL-USD"
]
```

#### Get Model Information
```
GET /models/{model_key}
```

##### Path Parameters
- `model_key` (string): Model identifier (e.g., "lstm_attention_BTC-USD")

##### Response
```json
{
  "model_type": "lstm_attention",
  "input_size": 20,
  "output_size": 1,
  "total_parameters": 33545,
  "trainable_parameters": 33545,
  "last_updated": "2023-12-01T12:00:00Z"
}
```

#### Status Codes
- `200 OK`: Model information retrieved successfully
- `404 Not Found`: Model not found

---

### Predictions

Generate cryptocurrency price predictions.

#### Make Prediction
```
POST /predict
```

##### Request Body
```json
{
  "ticker": "BTC-USD",
  "days": 5,
  "model_type": "lstm_attention",
  "confidence_interval": true,
  "confidence_level": 0.95
}
```

##### Request Parameters
- `ticker` (string, required): Cryptocurrency ticker (e.g., "BTC-USD")
- `days` (integer, required): Number of days to predict (1-30)
- `model_type` (string, required): Model type to use
- `confidence_interval` (boolean, optional): Include confidence intervals (default: false)
- `confidence_level` (float, optional): Confidence level (0.8-0.99, default: 0.95)

##### Response
```json
{
  "ticker": "BTC-USD",
  "predictions": [43250.25, 43375.80, 43512.30, 43648.90, 43785.45],
  "dates": ["2023-12-02", "2023-12-03", "2023-12-04", "2023-12-05", "2023-12-06"],
  "model_type": "lstm_attention",
  "confidence_intervals": [
    {
      "lower": 42890.50,
      "upper": 43610.00
    },
    {
      "lower": 43016.05,
      "upper": 43735.55
    },
    {
      "lower": 43141.60,
      "upper": 43861.10
    },
    {
      "lower": 43267.15,
      "upper": 43986.65
    },
    {
      "lower": 43392.70,
      "44112.20
    }
  ],
  "metrics": {
    "MAE": 125.50,
    "RMSE": 180.25,
    "MAPE": 0.0023,
    "Directional Accuracy": 0.65
  }
}
```

#### Status Codes
- `200 OK`: Prediction generated successfully
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Model not found
- `500 Internal Server Error`: Prediction failed

---

### Model Management

Manage model lifecycle including retraining and updates.

#### Retrain Model
```
POST /models/{model_key}/retrain
```

##### Path Parameters
- `model_key` (string): Model identifier

##### Request Body (Optional)
```json
{
  "epochs": 100,
  "learning_rate": 0.001,
  "batch_size": 32
}
```

##### Response
```json
{
  "message": "Model retraining started",
  "model_key": "lstm_attention_BTC-USD",
  "status": "in_progress",
  "estimated_time": "15 minutes"
}
```

#### Get Retraining Status
```
GET /models/{model_key}/retrain_status
```

##### Response
```json
{
  "model_key": "lstm_attention_BTC-USD",
  "status": "completed",
  "progress": 100,
  "metrics": {
    "MAE": 120.30,
    "RMSE": 175.45,
    "MAPE": 0.0021
  },
  "completed_at": "2023-12-01T12:30:00Z"
}
```

#### Status Codes
- `200 OK`: Request processed successfully
- `404 Not Found`: Model not found
- `409 Conflict`: Retraining already in progress

---

## Request/Response Formats

### Common Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-----------|
| `ticker` | string | Yes | Cryptocurrency ticker (e.g., "BTC-USD") |
| `model_type` | string | Yes | Model type ("lstm_attention", "transformer", "gru_attention", "informer", "ensemble") |
| `days` | integer | Yes | Number of days to predict (1-30) |
| `confidence_interval` | boolean | No | Include confidence intervals (default: false) |
| `confidence_level` | float | No | Confidence level (0.8-0.99, default: 0.95) |

### Common Response Fields

| Field | Type | Description |
|-------|------|-----------|
| `ticker` | string | Cryptocurrency ticker |
| `predictions` | array[float] | Predicted prices |
| `dates` | array[string] | Prediction dates (YYYY-MM-DD) |
| `model_type` | string | Model type used |
| `confidence_intervals` | array[object] | Confidence intervals (if requested) |
| `metrics` | object | Evaluation metrics (if available) |

### Confidence Interval Object

```json
{
  "lower": 42890.50,
  "upper": 43610.00
}
```

### Metrics Object

```json
{
  "MAE": 125.50,
  "MSE": 32490.06,
  "RMSE": 180.25,
  "MAPE": 0.0023,
  "Directional Accuracy": 0.65,
  "MASE": 0.78,
  "Sharpe Ratio": 1.23,
  "Max Drawdown": 0.15,
  "Information Ratio": 0.45,
  "Hit Rate": 0.58,
  "Average Return": 0.0012,
  "Error Volatility": 0.0234
}
```

## Error Codes

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 OK | Request successful |
| 400 Bad Request | Invalid parameters |
| 401 Unauthorized | Authentication required |
| 404 Not Found | Resource not found |
| 409 Conflict | Resource conflict |
| 422 Unprocessable Entity | Invalid request format |
| 429 Too Many Requests | Rate limit exceeded |
| 500 Internal Server Error | Server error |
| 503 Service Unavailable | Service down |

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_TICKER",
    "message": "Invalid cryptocurrency ticker: INVALID_TICKER",
    "details": {
      "valid_tickers": ["BTC-USD", "ETH-USD", "BNB-USD"],
      "requested": "INVALID_TICKER"
    }
  }
}
```

### Common Error Codes

| Code | Message | Solution |
|------|---------|----------|
| `INVALID_TICKER` | Invalid cryptocurrency ticker | Use valid Yahoo Finance ticker |
| `MODEL_NOT_FOUND` | Model not found | Check available models with GET /models |
| `INVALID_DAYS` | Invalid number of days | Use value between 1 and 30 |
| `MODEL_TYPE_REQUIRED` | Model type is required | Specify model_type in request |
| `RETRAINING_IN_PROGRESS` | Model is already being retrained | Wait for current retraining to complete |

## Rate Limiting

### Limits

- **Requests per minute**: 100
- **Requests per hour**: 1000
- **Concurrent requests**: 10

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1694109200
```

### Handling Rate Limits

```python
import time
import requests
from requests.exceptions import HTTPError

def make_request_with_retry(url, payload, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except HTTPError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get('Retry-After', 60))
                time.sleep(retry_after)
                if attempt == max_retries - 1:
                    raise
                continue
            raise

# Usage
response = make_request_with_retry(
    "http://localhost:8000/predict",
    {"ticker": "BTC-USD", "days": 5, "model_type": "lstm_attention"}
)
```

## Examples

### Basic Prediction

```python
import requests

# Make a prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "ticker": "BTC-USD",
        "days": 5,
        "model_type": "lstm_attention"
    }
)

result = response.json()
print(f"Predictions: {result['predictions']}")
print(f"Dates: {result['dates']}")
```

### Prediction with Confidence Intervals

```python
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "ticker": "ETH-USD",
        "days": 7,
        "model_type": "transformer",
        "confidence_interval": true,
        "confidence_level": 0.99
    }
)

result = response.json()
for i, (pred, date, ci) in enumerate(zip(result['predictions'], result['dates'], result['confidence_intervals'])):
    print(f"Day {i+1} ({date}): {pred} [{ci['lower']}, {ci['upper']}]")
```

### Batch Predictions

```python
cryptocurrencies = ["BTC-USD", "ETH-USD", "BNB-USD"]
results = {}

for crypto in cryptocurrencies:
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "ticker": crypto,
            "days": 5,
            "model_type": "lstm_attention"
        }
    )
    results[crypto] = response.json()

# Display results
for crypto, result in results.items():
    print(f"{crypto}: {result['predictions']}")
```

### Model Information

```python
# Get available models
models_response = requests.get("http://localhost:8000/models")
models = models_response.json()

# Get detailed info for each model
for model in models:
    info_response = requests.get(f"http://localhost:8000/models/{model}")
    info = info_response.json()
    print(f"{model}:")
    print(f"  Type: {info['model_type']}")
    print(f"  Parameters: {info['total_parameters']}")
    print(f"  Last Updated: {info['last_updated']}")
```

### Error Handling

```python
import requests
from requests.exceptions import RequestException

def safe_predict(ticker, days, model_type):
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={
                "ticker": ticker,
                "days": days,
                "model_type": model_type
            }
        )
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        print(f"Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            error_data = e.response.json()
            print(f"Error details: {error_data.get('error', {}).get('message', 'Unknown error')}")
        return None

# Usage
result = safe_predict("BTC-USD", 5, "lstm_attention")
if result:
    print(f"Predictions: {result['predictions']}")
```

### Streaming Predictions

```python
import sseclient
import requests

def stream_predictions(ticker, model_type):
    response = requests.post(
        "http://localhost:8000/predict/stream",
        json={
            "ticker": ticker,
            "days": 1,
            "model_type": model_type
        },
        stream=True
    )
    
    client = sseclient.SSEClient(response)
    for event in client.events():
        if event.event == 'prediction':
            data = json.loads(event.data)
            print(f"New prediction: {data}")

# Usage
stream_predictions("BTC-USD", "lstm_attention")
```

### WebSocket Connection

```python
import websocket
import json

def websocket_predictions():
    ws = websocket.create_connection("ws://localhost:8000/ws")
    
    # Subscribe to predictions
    ws.send(json.dumps({
        "action": "subscribe",
        "ticker": "BTC-USD"
    }))
    
    # Receive predictions
    while True:
        result = ws.recv()
        data = json.loads(result)
        print(f"Prediction: {data}")

# Usage
websocket_predictions()
```

## SDK Examples

### Python SDK

```python
from crypto_forecast_client import CryptoForecastClient

# Initialize client
client = CryptoForecastClient(base_url="http://localhost:8000")

# Make prediction
prediction = client.predict(
    ticker="BTC-USD",
    days=5,
    model_type="lstm_attention",
    confidence_interval=True
)

print(f"Prediction: {prediction.predictions}")
```

### JavaScript SDK

```javascript
import { CryptoForecastClient } from 'crypto-forecast-sdk';

// Initialize client
const client = new CryptoForecastClient('http://localhost:8000');

// Make prediction
const prediction = await client.predict({
    ticker: 'BTC-USD',
    days: 5,
    modelType: 'lstm_attention',
    confidenceInterval: true
});

console.log('Predictions:', prediction.predictions);
```

## Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t crypto-forecast-api .

# Run container
docker run -p 8000:8000 crypto-forecast-api
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crypto-forecast-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: crypto-forecast-api
  template:
    metadata:
      labels:
        app: crypto-forecast-api
    spec:
      containers:
      - name: api
        image: crypto-forecast:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/models"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

### Environment Variables

```bash
# Configuration
export MODEL_PATH=/app/models
export LOG_LEVEL=INFO
export MAX_WORKERS=4
export RATE_LIMIT=100
```

## Monitoring

### Health Monitoring

```python
import requests
import time

def monitor_api_health():
    while True:
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print(f"API is healthy: {response.json()}")
            else:
                print(f"API unhealthy: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"API error: {e}")
        
        time.sleep(60)  # Check every minute

# Usage
monitor_api_health()
```

### Performance Metrics

```python
import psutil
import torch

def get_system_metrics():
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    # GPU usage (if available)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_utilization = torch.cuda.utilization()
    else:
        gpu_memory = 0
        gpu_utilization = 0
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "gpu_memory_gb": gpu_memory,
        "gpu_utilization": gpu_utilization
    }

# Usage
metrics = get_system_metrics()
print(f"System Metrics: {metrics}")
```

## Changelog

### Version 1.0.0 (2023-12-01)
- Initial API release
- Basic prediction endpoints
- Model management endpoints
- Health check functionality

### Version 1.1.0 (2023-12-15)
- Added confidence intervals
- Streaming predictions support
- Enhanced error handling
- Rate limiting implementation

### Version 1.2.0 (2024-01-01)
- WebSocket support
- Batch prediction endpoint
- Performance optimizations
- Enhanced monitoring

## Support

For API support and questions:

- **Documentation**: [API Reference](api-reference.md)
- **Issues**: [GitHub Issues](https://github.com/Tushar-Siddik/crypto_forecasting/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Tushar-Siddik/crypto_forecasting/discussions)
- **Email**: 

---

*Last updated: January 15, 2026*