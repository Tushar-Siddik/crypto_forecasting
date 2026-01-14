# config/config.py
import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DataConfig:
    """Data configuration"""
    tickers: List[str] = None
    start_date: str = "2018-01-01"
    end_date: str = None
    sequence_length: int = 60
    prediction_horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 64
    data_dir: str = "data/raw"
    
    def __post_init__(self):
        if self.tickers is None:
            self.tickers = ["BTC-USD", "ETH-USD", "BNB-USD"]
        if self.end_date is None:
            from datetime import datetime
            self.end_date = datetime.now().strftime("%Y-%m-%d")

@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str = "lstm_attention"  # Options: lstm, lstm_attention, transformer, ensemble
    input_size: Optional[int] = None  # Will be set dynamically
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    attention_heads: int = 8
    d_model: int = 128
    num_encoder_layers: int = 3
    dim_feedforward: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 15
    save_dir: str = "models/saved"

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    metrics: List[str] = None
    plot_predictions: bool = True
    plot_residuals: bool = True
    save_results: bool = True
    results_dir: str = "results"
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["MAE", "RMSE", "MAPE", "Directional Accuracy", "MASE", "Sharpe Ratio"]

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"
    model_refresh_interval: int = 24  # hours

@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = None
    model: ModelConfig = None
    evaluation: EvaluationConfig = None
    deployment: DeploymentConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.deployment is None:
            self.deployment = DeploymentConfig()
        
        # Create directories
        os.makedirs(self.data.data_dir, exist_ok=True)
        os.makedirs(self.model.save_dir, exist_ok=True)
        os.makedirs(self.evaluation.results_dir, exist_ok=True)