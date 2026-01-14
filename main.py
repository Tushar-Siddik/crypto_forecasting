# main.py
import os
import sys
import argparse
import logging
from datetime import datetime
import json
import torch

from pathlib import Path

project_root = Path.cwd().parent
sys.path.append(str(project_root))

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.config import Config
from data.data_loader import CryptoDataLoader
from data.feature_engineering import FeatureEngineer
from data.preprocessor import DataPreprocessor
from models.lstm_attention import LSTMAttention, GRUAttention
from models.transformer import TransformerModel, InformerModel
from models.ensemble import EnsembleModel
from training.trainer import ModelTrainer
from training.hyperparameter_tuning import HyperparameterTuner
from evaluation.metrics import CryptoModelEvaluator
from evaluation.visualizer import ModelVisualizer
from utils import logger
from utils.logger import setup_logger
from utils.helpers import set_seed, save_results, create_timestamp

def main():
    """Main function to run the cryptocurrency forecasting system"""
    parser = argparse.ArgumentParser(description='Cryptocurrency Forecasting System')
    parser.add_argument('--config', type=str, default='config/config.json', help='Configuration file path')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'tune', 'evaluate', 'deploy'],
                       help='Operation mode')
    parser.add_argument('--ticker', type=str, default='BTC-USD', help='Cryptocurrency ticker')
    parser.add_argument('--model', type=str, default='lstm_attention', 
                       choices=['lstm_attention', 'gru_attention', 'transformer', 'informer', 'ensemble'],
                       help='Model type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger('crypto_forecasting', level=log_level)
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = Config()
    
    # Override config with command line arguments
    config.data.tickers = [args.ticker]
    config.model.model_type = args.model
    
    # Set device
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Run based on mode
    if args.mode == 'train':
        train_model(config, device)
    elif args.mode == 'tune':
        tune_hyperparameters(config, device)
    elif args.mode == 'evaluate':
        evaluate_model(config, device)
    elif args.mode == 'deploy':
        deploy_model(config, device)

def train_model(config: Config, device: torch.device) -> None:
    """Train a model"""
    logger.info("Starting model training...")
    
    # Initialize components
    loader = CryptoDataLoader(config.data.data_dir)
    feature_engineer = FeatureEngineer()
    preprocessor = DataPreprocessor()
    
    # Load data
    ticker = config.data.tickers[0]
    data = loader.get_latest_data(ticker, days=730)
    
    if data.empty:
        logger.error(f"No data found for {ticker}")
        return
    
    # Feature engineering
    logger.info("Performing feature engineering...")
    data_with_features = feature_engineer.add_technical_indicators(data)
    
    # Select features
    selected_features = feature_engineer.select_features(
        data_with_features, 
        method='correlation', 
        top_k=50
    )
    
    # Create sequences
    X, y = feature_engineer.create_sequences(
        data_with_features, 
        config.data.sequence_length,
        config.data.prediction_horizon
    )
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X, y, 
        config.data.train_ratio, 
        config.data.val_ratio
    )
    
    # Scale data
    X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled = preprocessor.scale_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Create data loaders
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled), 
        torch.FloatTensor(y_train_scaled)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled), 
        torch.FloatTensor(y_val_scaled)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled), 
        torch.FloatTensor(y_test_scaled)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.data.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.data.batch_size, shuffle=False)
    
    # Create model
    input_size = len(selected_features)
    
    if config.model.model_type == 'lstm_attention':
        model = LSTMAttention(
            input_size=input_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            output_size=1,
            dropout=config.model.dropout,
            bidirectional=config.model.bidirectional,
            attention_heads=config.model.attention_heads
        )
    elif config.model.model_type == 'gru_attention':
        model = GRUAttention(
            input_size=input_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            output_size=1,
            dropout=config.model.dropout,
            bidirectional=config.model.bidirectional,
            attention_heads=config.model.attention_heads
        )
    elif config.model.model_type == 'transformer':
        model = TransformerModel(
            input_size=input_size,
            d_model=config.model.d_model,
            nhead=config.model.nhead,
            num_encoder_layers=config.model.num_encoder_layers,
            dim_feedforward=config.model.dim_feedforward,
            output_size=1,
            dropout=config.model.dropout
        )
    elif config.model.model_type == 'informer':
        model = InformerModel(
            input_size=input_size,
            d_model=config.model.d_model,
            nhead=config.model.nhead,
            num_encoder_layers=config.model.num_encoder_layers,
            num_decoder_layers=2,
            dim_feedforward=config.model.dim_feedforward,
            output_size=1,
            dropout=config.model.dropout
        )
    else:
        logger.error(f"Unknown model type: {config.model.model_type}")
        return
    
    logger.info(f"Model created: {model.get_model_info()}")
    
    # Train model
    trainer = ModelTrainer(model, device)
    
    timestamp = create_timestamp()
    save_path = os.path.join(
        config.model.save_dir, 
        f"{config.model.model_type}_{ticker}_{timestamp}.pth"
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.model.epochs,
        lr=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
        patience=config.model.patience,
        save_path=save_path
    )
    
    # Evaluate model
    metrics, predictions, actuals = trainer.evaluate(test_loader, preprocessor.target_scaler)
    
    # Save results
    results = {
        'model_type': config.model.model_type,
        'ticker': ticker,
        'timestamp': timestamp,
        'model_info': model.get_model_info(),
        'training_history': history,
        'metrics': metrics,
        'config': {
            'data': config.data.__dict__,
            'model': config.model.__dict__
        }
    }
    
    results_path = os.path.join(
        config.evaluation.results_dir,
        f"{config.model.model_type}_{ticker}_{timestamp}.json"
    )
    save_results(results, results_path)
    
    # Save components
    model_save_dir = os.path.join(config.model.save_dir, f"{config.model.model_type}_{ticker}_{timestamp}")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Save model
    model.save_model(os.path.join(model_save_dir, "model.pth"))
    
    # Save configuration
    with open(os.path.join(model_save_dir, "config.json"), 'w') as f:
        json.dump(model.get_model_info(), f, indent=4)
    
    # Save scalers and feature columns
    preprocessor.save_scalers(model_save_dir)
    
    # Plot results
    visualizer = ModelVisualizer()
    
    # Plot training history
    trainer.plot_training_history(
        save_path=os.path.join(config.evaluation.results_dir, f"training_history_{timestamp}.png")
    )
    
    # Plot predictions
    trainer.plot_predictions(
        actuals, predictions,
        save_path=os.path.join(config.evaluation.results_dir, f"predictions_{timestamp}.png")
    )
    
    # Plot residuals
    trainer.plot_residuals(
        actuals, predictions,
        save_path=os.path.join(config.evaluation.results_dir, f"residuals_{timestamp}.png")
    )
    
    logger.info("Training completed successfully!")

def tune_hyperparameters(config: Config, device: torch.device) -> None:
    """Tune hyperparameters"""
    logger.info("Starting hyperparameter tuning...")
    
    # This would implement hyperparameter tuning
    # For brevity, we'll just log the action
    logger.info("Hyperparameter tuning not yet implemented")

def evaluate_model(config: Config, device: torch.device) -> None:
    """Evaluate a trained model"""
    logger.info("Starting model evaluation...")
    
    # This would implement model evaluation
    # For brevity, we'll just log the action
    logger.info("Model evaluation not yet implemented")

def deploy_model(config: Config, device: torch.device) -> None:
    """Deploy model as API"""
    logger.info("Starting model deployment...")
    
    # Import and run the API
    from deployment.api import app
    import uvicorn
    
    uvicorn.run(
        app, 
        host=config.deployment.host,
        port=config.deployment.port,
        reload=config.deployment.reload,
        log_level=config.deployment.log_level
    )

if __name__ == "__main__":
    main()