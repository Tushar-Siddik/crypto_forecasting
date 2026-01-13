# Cryptocurrency Price Forecasting System

This project implements a comprehensive time series forecasting system for cryptocurrency prices using LSTM, GRU, and Transformer models. The system includes data collection, feature engineering, model training, evaluation, and deployment.

## Features

- Data collection from Yahoo Finance API
- Advanced feature engineering with technical indicators
- Multiple model architectures (LSTM, GRU, Transformer)
- Comprehensive evaluation metrics
- RESTful API for real-time predictions
- Support for continuous learning and model updates

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Tushar-Siddik/crypto_forecasting.git

cd crypto_forecasting
```

2. Create a virtual environment:

```bash
python -m venv venv

source venv/bin/activate  

# On Windows: 
venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

To train a model for Bitcoin (BTC-USD) using LSTM:

```bash
python main.py --ticker BTC-USD --model lstm --train --epochs 50
```

To train a Transformer model for Ethereum (ETH-USD):

```bash
python main.py --ticker ETH-USD --model transformer --train --epochs 50
```

### Evaluating a Model

To evaluate a trained model:

```bash
python main.py --ticker BTC-USD --model lstm --evaluate
```

### Deploying the API

To deploy the model as a REST API:

```bash
python main.py --ticker BTC-USD --model lstm --deploy
```

The API will be available at `http://localhost:8000`.

### API Usage

Once the API is running, you can make predictions using the following endpoint:

```python
import requests

# Make a prediction for Bitcoin for the next 5 days
response = requests.post(
    "http://localhost:8000/predict",
    json={"ticker": "BTC-USD", "days": 5}
)

print(response.json())
```

## Project Structure

```
crypto_forecasting/
├── data/
│   └── raw/                    # Raw data files
├── models/                     # Saved model files
├── notebooks/                  # Jupyter notebooks for exploration
├── src/
│   ├── data/
│   │   ├── data_loader.py      # Data collection module
│   │   └── feature_engineering.py  # Feature engineering module
│   ├── models/
│   │   ├── lstm.py             # LSTM/GRU model implementation
│   │   ├── transformer.py      # Transformer model implementation
│   │   └── train.py            # Training and evaluation logic
│   ├── evaluation/
│   │   └── metrics.py          # Evaluation metrics
│   └── deployment/
│       └── api.py              # FastAPI deployment
├── main.py                     # Main script to run the project
├── requirements.txt            # Required packages
└── README.md                   # This file
```

## Model Architecture

### LSTM/GRU Model

The LSTM/GRU model consists of:
- Input layer
- One or more LSTM/GRU layers with dropout
- Fully connected output layer

### Transformer Model

The Transformer model consists of:
- Input projection layer
- Positional encoding
- Multiple Transformer encoder layers
- Fully connected output layer

## Feature Engineering

The system includes a comprehensive feature engineering module that creates:
- Momentum indicators (RSI, Stochastic Oscillator)
- Trend indicators (MACD, ADX, CCI)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators (OBV, VWAP)
- Price-based features (price change, log returns, volatility)
- Time-based features (day of week, month, quarter)

## Evaluation Metrics

The system evaluates models using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Directional Accuracy
- Mean Absolute Scaled Error (MASE)

## Future Improvements

- Implement ensemble methods
- Add support for probabilistic forecasting
- Implement online learning capabilities
- Add reinforcement learning for trading strategies
- Improve feature selection with automated methods
- Add support for more data sources

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE)

## Example Usage Script

Let's create an example script that demonstrates how to use the project:

```python
# example_usage.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_loader import CryptoDataLoader
from src.data.feature_engineering import FeatureEngineer
from src.models.train import TimeSeriesTrainer
from src.evaluation.metrics import evaluate_model

def example_data_collection():
    """Example of data collection"""
    print("=== Data Collection Example ===")
    
    # Initialize data loader
    loader = CryptoDataLoader()
    
    # Fetch Bitcoin data for the last year
    btc_data = loader.get_latest_data('BTC-USD', days=365)
    
    # Display the first few rows
    print("Bitcoin data (first 5 rows):")
    print(btc_data.head())
    
    # Plot the closing price
    plt.figure(figsize=(12, 6))
    plt.plot(btc_data['Close'])
    plt.title('Bitcoin Closing Price (Last Year)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.show()
    
    return btc_data

def example_feature_engineering(data):
    """Example of feature engineering"""
    print("\n=== Feature Engineering Example ===")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Add technical indicators
    data_with_features = engineer.add_technical_indicators(data)
    
    # Display the new features
    print("Data with technical indicators (first 5 rows, selected columns):")
    print(data_with_features[['Close', 'rsi', 'macd', 'bollinger_hband', 'bollinger_lband']].head())
    
    # Plot some indicators
    plt.figure(figsize=(12, 10))
    
    # Plot price and Bollinger Bands
    plt.subplot(2, 1, 1)
    plt.plot(data_with_features['Close'], label='Close Price')
    plt.plot(data_with_features['bollinger_hband'], label='Upper Bollinger Band')
    plt.plot(data_with_features['bollinger_lband'], label='Lower Bollinger Band')
    plt.fill_between(data_with_features.index, 
                     data_with_features['bollinger_hband'], 
                     data_with_features['bollinger_lband'], 
                     color='gray', alpha=0.2)
    plt.title('Bitcoin Price with Bollinger Bands')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    # Plot RSI
    plt.subplot(2, 1, 2)
    plt.plot(data_with_features['rsi'])
    plt.axhline(y=70, color='r', linestyle='--')
    plt.axhline(y=30, color='g', linestyle='--')
    plt.title('RSI Indicator')
    plt.ylabel('RSI')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return data_with_features

def example_model_training():
    """Example of model training"""
    print("\n=== Model Training Example ===")
    
    # Initialize trainer for LSTM model
    trainer = TimeSeriesTrainer(model_type='lstm')
    
    # Prepare data
    print("Preparing data...")
    train_loader, val_loader, test_loader, feature_scaler, target_scaler, feature_cols = trainer.prepare_data(
        ticker='BTC-USD',
        sequence_length=60
    )
    
    # Build model
    print("Building LSTM model...")
    model = trainer.build_model(
        input_size=len(feature_cols),
        hidden_size=64,
        num_layers=2,
        output_size=1,
        dropout=0.2
    )
    
    # Train model (using fewer epochs for demonstration)
    print("Training model...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,  # Using fewer epochs for demonstration
        lr=0.001,
        patience=5
    )
    
    # Plot training history
    trainer.plot_history()
    
    # Evaluate model
    print("Evaluating model...")
    metrics, predictions, actuals = trainer.evaluate(test_loader, target_scaler)
    
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot predictions
    trainer.plot_predictions(actuals, predictions, n=100)
    
    return trainer, metrics

def example_model_comparison():
    """Example of comparing different models"""
    print("\n=== Model Comparison Example ===")
    
    # Cryptocurrencies to test
    tickers = ['BTC-USD', 'ETH-USD']
    
    # Models to compare
    models = ['lstm', 'transformer']
    
    # Results dictionary
    results = {}
    
    for ticker in tickers:
        results[ticker] = {}
        
        for model_type in models:
            print(f"\nTraining {model_type} model for {ticker}...")
            
            # Initialize trainer
            trainer = TimeSeriesTrainer(model_type=model_type)
            
            # Prepare data
            train_loader, val_loader, test_loader, feature_scaler, target_scaler, feature_cols = trainer.prepare_data(
                ticker=ticker,
                sequence_length=60
            )
            
            # Build model
            if model_type == 'lstm':
                model = trainer.build_model(
                    input_size=len(feature_cols),
                    hidden_size=64,
                    num_layers=2,
                    output_size=1,
                    dropout=0.2
                )
            else:  # transformer
                model = trainer.build_model(
                    input_size=len(feature_cols),
                    d_model=64,
                    nhead=4,
                    num_encoder_layers=2,
                    dim_feedforward=128,
                    output_size=1,
                    dropout=0.1
                )
            
            # Train model (using fewer epochs for demonstration)
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=10,  # Using fewer epochs for demonstration
                lr=0.001,
                patience=5
            )
            
            # Evaluate model
            metrics, _, _ = trainer.evaluate(test_loader, target_scaler)
            
            # Store results
            results[ticker][model_type] = metrics
    
    # Display results
    print("\n=== Model Comparison Results ===")
    for ticker in tickers:
        print(f"\n{ticker}:")
        for model_type in models:
            print(f"  {model_type}:")
            for metric, value in results[ticker][model_type].items():
                print(f"    {metric}: {value:.4f}")

def main():
    """Main function to run all examples"""
    # Example 1: Data collection
    data = example_data_collection()
    
    # Example 2: Feature engineering
    data_with_features = example_feature_engineering(data)
    
    # Example 3: Model training
    trainer, metrics = example_model_training()
    
    # Example 4: Model comparison
    example_model_comparison()

if __name__ == "__main__":
    main()
```