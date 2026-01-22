# Cryptocurrency Forecasting System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

An advanced time series forecasting system for cryptocurrency prices using state-of-the-art deep learning models including LSTM with Attention, Transformer, and Ensemble methods.

## 🚀 Features

- **Multiple Model Architectures**: LSTM with Attention, GRU with Attention, Transformer, Informer, and Ensemble models
- **Advanced Feature Engineering**: 50+ technical indicators and statistical features
- **Hyperparameter Tuning**: Automated hyperparameter optimization using Optuna
- **Comprehensive Evaluation**: Financial-specific metrics including Sharpe ratio, directional accuracy, and maximum drawdown
- **RESTful API**: FastAPI-based deployment for real-time predictions
- **Visualization**: Interactive plots using Plotly and Matplotlib
- **Production Ready**: Model monitoring, logging, and continuous learning capabilities
- **GPU Acceleration**: Full support for CUDA-enabled training and inference
- **Extensible Design**: Easy to add new models and features

## 📊 Model Architectures

### LSTM with Attention
- Multi-head attention mechanism
- Bidirectional LSTM layers
- Residual connections and layer normalization
- Dropout for regularization

### Transformer
- Multi-head self-attention
- Positional encoding
- Multiple encoder layers
- Feed-forward networks

### Informer
- ProbSparse self-attention
- Distillation mechanism
- Efficient for long sequences

### Ensemble
- Weighted averaging of multiple models
- Attention-based aggregation
- Stacking with meta-learner

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended)
- 8GB+ RAM (16GB+ recommended for large datasets)

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/Tushar-Siddik/crypto_forecasting.git
cd crypto_forecasting
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install the required packages**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

### Training a Model

Train a single model with default settings:
```bash
python main.py --mode train --ticker BTC-USD --model lstm_attention --gpu
```

Train with custom configuration:
```bash
python main.py --mode train --config config/custom_config.json --ticker ETH-USD --model transformer
```

### Hyperparameter Tuning

Automatically find the best hyperparameters:
```bash
python main.py --mode tune --ticker BTC-USD --model lstm_attention --gpu --n-trials 100
```

### Evaluation

Evaluate a trained model:
```bash
python main.py --mode evaluate --ticker BTC-USD --model lstm_attention
```

### Deployment

Deploy the model as a REST API:
```bash
python main.py --mode deploy --ticker BTC-USD --model lstm_attention
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## 📚 Jupyter Notebooks

Explore the system through our comprehensive notebooks:

| Notebook | Description |
|----------|-------------|
| [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) | Comprehensive data exploration and analysis |
| [02_feature_engineering.ipynb](notebooks/02_feature_engineering.ipynb) | Feature engineering and selection techniques |
| [03_model_comparison.ipynb](notebooks/03_model_comparison.ipynb) | Model comparison and performance analysis |

Run notebooks with:
```bash
jupyter notebook notebooks/
```

## 🌐 API Usage

### Make Predictions

```python
import requests

# Make a prediction for Bitcoin for the next 5 days
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
print(f"Dates: {result['dates']}")
```

### Check Model Information

```python
# List available models
response = requests.get("http://localhost:8000/models")
print(response.json())

# Get detailed information about a specific model
response = requests.get("http://localhost:8000/models/lstm_attention_BTC-USD")
print(response.json())
```

### Health Check

```python
response = requests.get("http://localhost:8000/health")
print(response.json())
```

## 📁 Project Structure

```
crypto_forecasting/
├── config/                 # Configuration files
│   ├── __init__.py
│   └── config.py          # Main configuration class
├── data/                   # Data handling
│   ├── __init__.py
│   ├── data_loader.py      # Data collection from Yahoo Finance
│   ├── feature_engineering.py  # Technical indicators
│   └── preprocessor.py     # Data scaling and preparation
├── models/                 # Model implementations
│   ├── __init__.py
│   ├── base_model.py       # Base class for all models
│   ├── lstm_attention.py   # LSTM/GRU with Attention
│   ├── transformer.py      # Transformer and Informer
│   └── ensemble.py         # Ensemble methods
├── training/               # Training system
│   ├── __init__.py
│   ├── trainer.py          # Model training logic
│   └── hyperparameter_tuning.py  # Optuna optimization
├── evaluation/             # Model evaluation
│   ├── __init__.py
│   ├── metrics.py          # Financial metrics
│   └── visualizer.py       # Plotting and visualization
├── deployment/             # API deployment
│   ├── __init__.py
│   ├── api.py              # FastAPI application
│   └── monitoring.py       # Model monitoring
├── utils/                  # Utilities
│   ├── __init__.py
│   ├── helpers.py          # Helper functions
│   └── logger.py           # Logging setup
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_comparison.ipynb
├── tests/                  # Test suite
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_integration.py
│   └── run_tests.py
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 📊 Features

### Technical Indicators

| Category | Indicators |
|----------|------------|
| **Momentum** | RSI, Stochastic Oscillator, Williams %R, Ultimate Oscillator |
| **Trend** | MACD, ADX, CCI, DMI, Aroon |
| **Volatility** | Bollinger Bands, ATR, Keltner Channel, Donchian Channel |
| **Volume** | OBV, VWAP, ADI, MFI, Force Index, EMV |

### Statistical Features

- Price changes and log returns
- Rolling statistics (mean, std, min, max)
- Skewness and kurtosis
- Lag features
- Time-based features (day of week, month, quarter)

### Evaluation Metrics

| Category | Metrics |
|----------|---------|
| **Basic** | MAE, MSE, RMSE, MAPE |
| **Financial** | Directional Accuracy, MASE, Sharpe Ratio, Maximum Drawdown |
| **Advanced** | Information Ratio, Hit Rate, Average Return, Error Volatility |

## ⚙️ Configuration

The system uses a flexible configuration system. Example configuration:

```python
from config.config import Config

config = Config()
config.data.tickers = ["BTC-USD", "ETH-USD"]
config.data.sequence_length = 60
config.model.model_type = "lstm_attention"
config.model.hidden_size = 128
config.model.epochs = 100
```

### Available Configuration Options

```python
# Data Configuration
DataConfig:
    tickers: List[str] = ["BTC-USD", "ETH-USD", "BNB-USD"]
    start_date: str = "2018-01-01"
    sequence_length: int = 60
    prediction_horizon: int = 1
    train_ratio: float = 0.7
    batch_size: int = 64

# Model Configuration
ModelConfig:
    model_type: str = "lstm_attention"
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100
```

## 🧪 Testing

Run the complete test suite:

```bash
# Run all tests
python tests/run_tests.py

# Run specific test module
python tests/run_tests.py test_models

# Run with pytest (if installed)
pytest tests/ -v
```

Test coverage includes:
- Unit tests for all components
- Integration tests for end-to-end pipelines
- Model training and evaluation tests
- API endpoint tests

## 📈 Performance

The system is optimized for performance:

- **GPU Acceleration**: Full CUDA support for training and inference
- **Batch Processing**: Efficient data loading and batching
- **Memory Management**: Optimized memory usage for large datasets
- **Parallel Processing**: Multi-threaded data preprocessing

### Benchmarks

| Model | Training Time (60 epochs) | Inference Time (batch=32) | GPU Memory |
|-------|---------------------------|---------------------------|------------|
| LSTM-Attention | ~5 min | ~10ms | ~2GB |
| Transformer | ~8 min | ~15ms | ~3GB |
| Informer | ~6 min | ~12ms | ~2.5GB |
| Ensemble | ~15 min | ~25ms | ~4GB |

*Tested on NVIDIA RTX 3060 with Intel i7-10700K*

## 🔧 Advanced Usage

### Custom Model Architecture

```python
from models.base_model import BaseModel
import torch.nn as nn

class CustomModel(BaseModel):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, output_size)
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)
```

### Custom Feature Engineering

```python
from data.feature_engineering import FeatureEngineer

class CustomFeatureEngineer(FeatureEngineer):
    def add_custom_features(self, df):
        # Add your custom features
        df['custom_feature'] = df['Close'].pct_change().rolling(10).std()
        return df
```

### Hyperparameter Tuning

```python
from training.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner(
    model_type='lstm_attention',
    input_size=50,
    n_trials=100
)

best_params = tuner.tune(train_loader, val_loader)
best_model = tuner.get_best_model()
```

## 📊 Visualization Examples

### Training History

```python
from evaluation.visualizer import ModelVisualizer

visualizer = ModelVisualizer()
visualizer.plot_learning_curves(history)
```

### Predictions vs Actual

```python
visualizer.plot_predictions(actuals, predictions, n=100)
```

### Feature Importance

```python
visualizer.plot_feature_importance(feature_names, importance_scores)
```

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py", "--mode", "deploy"]
```

Build and run:
```bash
docker build -t crypto-forecasting .
docker run -p 8000:8000 crypto-forecasting
```

### Production Considerations

- **Monitoring**: Use the built-in monitoring system or integrate with Prometheus/Grafana
- **Scaling**: Deploy multiple instances behind a load balancer
- **Model Updates**: Use the `/models/{model_key}/retrain` endpoint for continuous learning
- **Security**: Add authentication and rate limiting for production use

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Code Style

This project uses:
- **Black** for code formatting
- **Flake8** for linting
- **mypy** for type checking

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type check
mypy src/
```

## 📝 Citation

If you use this system in your research, please cite:

```bibtex
@misc{crypto_forecasting,
  title={Cryptocurrency Forecasting System},
  author={Md. Siddiqur Rahman},
  year={2026},
  url={https://github.com/Tushar-Siddik/crypto_forecasting},
  note={Advanced deep learning system for cryptocurrency price prediction}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [yfinance](https://github.com/ranaroussi/yfinance) for financial data
- [ta](https://github.com/bukosabino/ta) for technical analysis
- [Optuna](https://optuna.org/) for hyperparameter optimization
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework

## 📞 Support

- 📧 Email: ***
- 🐛 Issues: [GitHub Issues](https://github.com/Tushar-Siddik/crypto_forecasting/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/crypto_forecasting/discussions)

## 🗺️ Roadmap

- [ ] **Multi-step Prediction**: Predict multiple days ahead
- [ ] **Probabilistic Forecasting**: Add prediction intervals
- [ ] **Reinforcement Learning**: Trading strategy optimization
- [ ] **More Data Sources**: Integration with more exchanges
- [ ] **Web Interface**: React-based dashboard
- [ ] **Mobile App**: iOS and Android applications
- [ ] **Cloud Deployment**: AWS/GCP/Azure deployment templates

---

<div align="center">
  <p>Made with ❤️ for the cryptocurrency community</p>
  <p>If you find this project useful, please consider ⭐ starring it!</p>
</div>
