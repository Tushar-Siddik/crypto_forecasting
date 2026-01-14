# Cryptocurrency Forecasting System - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Data Preparation](#data-preparation)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [API Usage](#api-usage)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Introduction

The Cryptocurrency Forecasting System is a comprehensive deep learning framework for predicting cryptocurrency prices. It supports multiple model architectures including LSTM with Attention, Transformer, and Ensemble methods, with advanced feature engineering and evaluation capabilities.

### Key Features

- **Multiple Model Architectures**: LSTM, GRU, Transformer, Informer, and Ensemble models
- **Advanced Feature Engineering**: 50+ technical indicators and statistical features
- **Comprehensive Evaluation**: Financial-specific metrics including Sharpe ratio and directional accuracy
- **RESTful API**: Easy deployment for real-time predictions
- **Extensible Design**: Easy to add new models and features

### Supported Cryptocurrencies

The system works with any cryptocurrency available on Yahoo Finance, including:
- Bitcoin (BTC-USD)
- Ethereum (ETH-USD)
- Binance Coin (BNB-USD)
- Cardano (ADA-USD)
- Solana (SOL-USD)
- And many more...

## Installation

### System Requirements

- Python 3.8 or higher
- 8GB+ RAM (16GB+ recommended for large datasets)
- CUDA-compatible GPU (optional but recommended for training)

### Step 1: Clone the Repository

```bash
git https://github.com/Tushar-Siddik/crypto_forecasting.git
cd crypto_forecasting
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda (optional)
conda create -n crypto_forecast python=3.9
conda activate crypto_forecast
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```python
import torch
import pandas as pd
import yfinance as ta

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Quick Start

### Basic Usage Example

```python
from src.data.data_loader import CryptoDataLoader
from src.data.feature_engineering import FeatureEngineer
from src.models.lstm_attention import LSTMAttention
from src.training.trainer import ModelTrainer

# Load data
loader = CryptoDataLoader()
data = loader.get_latest_data('BTC-USD', days=365)

# Feature engineering
engineer = FeatureEngineer()
data_with_features = engineer.add_technical_indicators(data)

# Create sequences
X, y = engineer.create_sequences(data_with_features, sequence_length=30)

# Split and scale data
X_train, X_val, X_test, y_train, y_val, y_test = engineer.split_data(X, y)
X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled = engineer.scale_data(X_train, X_val, X_test, y_train, y_val, y_test)

# Create model
model = LSTMAttention(input_size=len(engineer.feature_columns), hidden_size=64, num_layers=2)

# Train model
trainer = ModelTrainer(model)
history = trainer.train(train_loader, val_loader, epochs=50)

# Evaluate
metrics, predictions, actuals = trainer.evaluate(test_loader, engineer.target_scaler)
print(f"RMSE: {metrics['RMSE']:.4f}")
```

### Command Line Usage

```bash
# Train a model
python main.py --mode train --ticker BTC-USD --model lstm_attention --gpu

# Tune hyperparameters
python main.py --mode tune --ticker ETH-USD --model transformer --n-trials 100

# Deploy API
python main.py --mode deploy --ticker BTC-USD --model lstm_attention
```

## Data Preparation

### Loading Data

#### From Yahoo Finance (Default)

```python
from src.data.data_loader import CryptoDataLoader

loader = CryptoDataLoader()
data = loader.get_latest_data('BTC-USD', days=365)
print(data.head())
```

#### From CSV File

```python
data = loader.load_saved_data('BTC-USD', '2022-01-01', '2022-12-31')
```

#### Multiple Cryptocurrencies

```python
tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD']
data_dict = loader.get_market_data(tickers, days=365)
```

### Data Exploration

```python
# Basic statistics
print(data.describe())

# Visualize price trends
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'])
plt.title('Bitcoin Price (Last Year)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.show()
```

### Feature Engineering

#### Automatic Feature Generation

```python
from src.data.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
data_with_features = engineer.add_technical_indicators(data)
print(f"Added {len(data_with_features.columns)} features")
print(data_with_features.columns.tolist())
```

#### Custom Features

```python
# Add custom features
data_with_features['Custom_MA'] = data['Close'].rolling(window=10).mean()
data_with_features['Price_Momentum'] = data['Close'].pct_change(5)
```

#### Feature Selection

```python
# Select top features by correlation
selected_features = engineer.select_features(
    data_with_features, 
    method='correlation', 
    top_k=20
)
print(f"Selected {len(selected_features)} features")
```

### Creating Sequences

```python
# Create sequences for time series modeling
X, y = engineer.create_sequences(
    data_with_features, 
    sequence_length=30,  # Use last 30 days to predict next day
    prediction_horizon=1
)

print(f"X shape: {X.shape}")  # (samples, sequence_length, features)
print(f"y shape: {y.shape}")  # (samples,)
```

### Data Splitting and Scaling

```python
from src.data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

# Scale data
X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled = preprocessor.scale_data(
    X_train, X_val, X_test, y_train, y_val, y_test
)
```

## Model Training

### Available Models

#### LSTM with Attention

```python
from src.models.lstm_attention import LSTMAttention

model = LSTMAttention(
    input_size=len(selected_features),
    hidden_size=128,
    num_layers=2,
    output_size=1,
    dropout=0.2,
    bidirectional=True,
    attention_heads=8
)
```

#### Transformer

```python
from src.models.transformer import TransformerModel

model = TransformerModel(
    input_size=len(selected_features),
    d_model=128,
    nhead=8,
    num_encoder_layers=3,
    dim_feedforward=256,
    output_size=1
)
```

#### Ensemble Model

```python
from src.models.ensemble import EnsembleModel

base_models = [lstm_model, transformer_model]
ensemble = EnsembleModel(
    models=base_models,
    input_size=len(selected_features),
    output_size=1,
    aggregation_method='weighted_average'
)
```

### Training Configuration

```python
from src.training.trainer import ModelTrainer

trainer = ModelTrainer(model, device='cuda')  # Use 'cpu' if no GPU

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    lr=0.001,
    weight_decay=1e-5,
    patience=15,
    early_stopping=True,
    scheduler_type='reduce_on_plateau',
    gradient_clipping=1.0
)
```

### Hyperparameter Tuning

```python
from src.training.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner(
    model_type='lstm_attention',
    input_size=len(selected_features),
    n_trials=100
)

best_params = tuner.tune(train_loader, val_loader)
best_model = tuner.get_best_model()
```

### Training with Custom Configuration

```python
# Create custom configuration
config = {
    'epochs': 50,
    'lr': 0.001,
    'batch_size': 32,
    'patience': 10,
    'scheduler_type': 'cosine'
}

# Train with custom config
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    **config
)
```

## Model Evaluation

### Basic Evaluation

```python
metrics, predictions, actuals = trainer.evaluate(test_loader, preprocessor.target_scaler)

print("Evaluation Metrics:")
for metric, value in metrics.items():
    if value is not None:
        print(f"  {metric}: {value:.4f}")
```

### Financial Metrics

The system includes specialized financial metrics:

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **Directional Accuracy**: Percentage of correct direction predictions
- **MASE**: Mean Absolute Scaled Error
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Information Ratio**: Active return relative to benchmark

### Visualization

```python
from src.evaluation.visualizer import ModelVisualizer

visualizer = ModelVisualizer()

# Plot predictions vs actual
visualizer.plot_predictions(actuals, predictions, n=100)

# Plot residuals
visualizer.plot_residuals(actuals, predictions)

# Plot training history
visualizer.plot_learning_curves(history)
```

### Model Comparison

```python
# Compare multiple models
results = {
    'LSTM': lstm_metrics,
    'Transformer': transformer_metrics,
    'Ensemble': ensemble_metrics
}

visualizer.plot_model_comparison(results)
```

## API Usage

### Starting the API

```bash
python main.py --mode deploy
```

The API will be available at `http://localhost:8000`

### Making Predictions

```python
import requests

# Make a prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "ticker": "BTC-USD",
        "days": 5,
        "model_type": "lstm_attention",
        "confidence_interval": True
    }
)

result = response.json()
print(f"Predictions: {result['predictions']}")
```

### API Endpoints

#### Get Available Models

```python
response = requests.get("http://localhost:8000/models")
models = response.json()
print(models)
```

#### Get Model Information

```python
response = requests.get("http://localhost:8000/models/lstm_attention_BTC-USD")
model_info = response.json()
print(model_info)
```

#### Health Check

```python
response = requests.get("http://localhost:8000/health")
health = response.json()
print(health)
```

### Batch Predictions

```python
# Predict for multiple cryptocurrencies
cryptocurrencies = ['BTC-USD', 'ETH-USD', 'BNB-USD']
results = {}

for crypto in cryptocurrencies:
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "ticker": crypto,
            "days": 7,
            "model_type": "lstm_attention"
        }
    )
    results[crypto] = response.json()
```

## Advanced Features

### Multi-step Prediction

```python
# Configure for multi-step prediction
X, y = engineer.create_sequences(
    data_with_features, 
    sequence_length=30,
    prediction_horizon=7  # Predict next 7 days
)

# Use a model that supports multi-step output
model = MultiStepTransformer(
    input_size=len(selected_features),
    output_size=7  # Predict 7 days at once
)
```

### Probabilistic Forecasting

```python
# Enable prediction intervals
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "ticker": "BTC-USD",
        "days": 5,
        "model_type": "lstm_attention",
        "confidence_interval": True,
        "confidence_level": 0.95
    }
)

result = response.json()
print(f"Predictions: {result['predictions']}")
print(f"Confidence Intervals: {result['confidence_intervals']}")
```

### Custom Model Architecture

```python
from src.models.base_model import BaseModel
import torch.nn as nn

class CustomModel(BaseModel):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, output_size)
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        return self.output(x)
```

### Custom Feature Engineering

```python
from src.data.feature_engineering import FeatureEngineer

class CustomFeatureEngineer(FeatureEngineer):
    def add_custom_features(self, df):
        # Add your custom features
        df['custom_feature_1'] = df['Close'].pct_change(5)
        df['custom_feature_2'] = df['Volume'].rolling(10).mean()
        return df
```

### Continuous Learning

```python
# Retrain model with new data
response = requests.post(
    "http://localhost:8000/models/lstm_attention_BTC-USD/retrain"
)
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

```python
# Reduce batch size
trainer = ModelTrainer(model, device='cuda')
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    batch_size=16  # Reduce from 32
)
```

#### Poor Performance

```python
# Try different hyperparameters
model = LSTMAttention(
    input_size=len(selected_features),
    hidden_size=256,  # Increase hidden size
    num_layers=3,     # Add more layers
    dropout=0.3       # Increase dropout
)
```

#### Slow Training

```python
# Use mixed precision training
model = model.half()  # Convert to half precision
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)
    scaler.scale(loss).backward()
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug prints
trainer = ModelTrainer(model, device='cpu')  # Use CPU for debugging
```

### Performance Optimization

```python
# Use DataParallel for multi-GPU
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Pin memory for faster data loading
for inputs, targets in train_loader:
    inputs = inputs.pin_memory().to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
```

## Best Practices

### Data Preparation

1. **Always scale your data** - Use MinMaxScaler or StandardScaler
2. **Handle missing values** - Drop or impute appropriately
3. **Remove outliers** - Use statistical methods or domain knowledge
4. **Use time-aware splits** - Never shuffle time series data

### Model Training

1. **Use early stopping** - Prevent overfitting
2. **Monitor validation loss** - Track both training and validation
3. **Use learning rate scheduling** - Improve convergence
4. **Apply gradient clipping** - Prevent exploding gradients

### Evaluation

1. **Use multiple metrics** - Don't rely on a single metric
2. **Consider financial metrics** - Sharpe ratio, drawdown, etc.
3. **Visualize results** - Plot predictions and residuals
4. **Test on out-of-sample data** - Ensure generalization

### Production

1. **Version control your models** - Track model versions and performance
2. **Monitor model drift** - Retrain when performance degrades
3. **Implement A/B testing** - Compare model versions in production
4. **Set up monitoring** - Track predictions and system health

### Code Organization

1. **Follow the project structure** - Keep code organized
2. **Write clean, documented code** - Use docstrings and comments
3. **Use type hints** - Improve code clarity and catch errors
4. **Write tests** - Ensure code correctness

## Examples and Tutorials

### Example 1: Bitcoin Price Prediction

```python
# Complete example for Bitcoin prediction
from src.data.data_loader import CryptoDataLoader
from src.data.feature_engineering import FeatureEngineer
from src.models.lstm_attention import LSTMAttention
from src.training.trainer import ModelTrainer

# Load and prepare data
loader = CryptoDataLoader()
data = loader.get_latest_data('BTC-USD', days=730)

engineer = FeatureEngineer()
data_with_features = engineer.add_technical_indicators(data)

# Create sequences
X, y = engineer.create_sequences(data_with_features, sequence_length=60)

# Split and scale
preprocessor = DataPreprocessor()
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled = preprocessor.scale_data(
    X_train, X_val, X_test, y_train, y_val, y_test
)

# Create and train model
model = LSTMAttention(
    input_size=len(engineer.feature_columns),
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)

trainer = ModelTrainer(model, device='cuda')
history = trainer.train(
    train_loader, val_loader, epochs=100
)

# Evaluate
metrics, predictions, actuals = trainer.evaluate(test_loader, preprocessor.target_scaler)
print(f"RMSE: {metrics['RMSE']:.4f}")
print(f"Directional Accuracy: {metrics['Directional Accuracy']:.4f}")
```

### Example 2: Multi-Cryptocurrency Comparison

```python
# Compare models across multiple cryptocurrencies
cryptocurrencies = ['BTC-USD', 'ETH-USD', 'BNB-USD']
results = {}

for crypto in cryptocurrencies:
    # Load data
    data = loader.get_latest_data(crypto, days=365)
    
    # Feature engineering
    data_with_features = engineer.add_technical_indicators(data)
    
    # Create sequences
    X, y = engineer.create_sequences(data_with_features, sequence_length=30)
    
    # Split and scale
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled = preprocessor.scale_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Train model
    model = LSTMAttention(input_size=len(engineer.feature_columns), hidden_size=64)
    trainer = ModelTrainer(model, device='cuda')
    trainer.train(train_loader, val_loader, epochs=50)
    
    # Evaluate
    metrics, _, _ = trainer.evaluate(test_loader, preprocessor.target_scaler)
    results[crypto] = metrics

# Compare results
for crypto, metrics in results.items():
    print(f"{crypto}: RMSE={metrics['RMSE']:.4f}, DirAcc={metrics['Directional Accuracy']:.4f}")
```

### Example 3: Custom Model Implementation

```python
# Implement a custom GRU model
from src.models.base_model import BaseModel
import torch.nn as nn

class CustomGRU(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__(input_size, output_size)
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        output, _ = self.gru(x)
        output = self.dropout(output[:, -1, :])  # Use last time step
        return self.fc(output)

# Use the custom model
model = CustomGRU(
    input_size=len(selected_features),
    hidden_size=128,
    num_layers=2,
    output_size=1
)

trainer = ModelTrainer(model, device='cuda')
history = trainer.train(train_loader, val_loader, epochs=50)
```

## Additional Resources

### Documentation

- [API Reference](docs/api.md)
- [Architecture Guide](docs/architecture.md)
- [Contributing Guide](CONTRIBUTING.md)

### Research Papers

- "LSTM: A Search Space Odyssey" (Hochreiter & Schmidhuber, 1997)
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (Zhou et al., 2021)

### Online Courses

- Coursera: "Sequence Models" by Andrew Ng
- Fast.ai: "Practical Deep Learning for Coders"
- Udacity: "Deep Learning Nanodegree"

### Communities

- [PyTorch Forums](https://discuss.pytorch.org/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/pytorch)
- [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/)

## Support

If you encounter issues or have questions:

1. Check the [troubleshooting section](#troubleshooting)
2. Search [existing issues](https://github.com/Tushar-Siddik/crypto_forecasting/issues)
3. Create a [new issue](https://github.com/Tushar-Siddik/crypto_forecasting/issues/new)
4. Join our [GitHub Discussions](https://github.com/Tushar-Siddik/crypto_forecasting/discussions)

---

Happy forecasting! 🚀