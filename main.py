# main.py
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_loader import CryptoDataLoader
from src.data.feature_engineering import FeatureEngineer
from src.models.train import TimeSeriesTrainer
from src.evaluation.metrics import evaluate_model

def main():
    parser = argparse.ArgumentParser(description='Crypto Price Forecasting')
    parser.add_argument('--ticker', type=str, default='BTC-USD', help='Cryptocurrency ticker')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'transformer'], help='Model type')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--sequence_length', type=int, default=60, help='Sequence length for time series')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--deploy', action='store_true', help='Deploy the model as API')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = TimeSeriesTrainer(model_type=args.model)
    
    # Prepare data
    print(f"Preparing data for {args.ticker}...")
    train_loader, val_loader, test_loader, feature_scaler, target_scaler, feature_cols = trainer.prepare_data(
        ticker=args.ticker,
        sequence_length=args.sequence_length
    )
    
    # Build model
    print(f"Building {args.model} model...")
    input_size = len(feature_cols)
    
    if args.model == 'lstm':
        model = trainer.build_model(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout=0.2,
            bidirectional=False,
            use_gru=False
        )
    else:  # transformer
        model = trainer.build_model(
            input_size=input_size,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=128,
            output_size=1,
            dropout=0.1
        )
    
    # Train model
    if args.train:
        print(f"Training {args.model} model...")
        save_path = os.path.join(args.save_dir, f"{args.ticker}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=0.001,
            patience=10,
            save_path=save_path
        )
        
        # Plot training history
        trainer.plot_history()
        
        print(f"Model saved to {save_path}")
    
    # Evaluate model
    if args.evaluate:
        print(f"Evaluating {args.model} model...")
        metrics, predictions, actuals = trainer.evaluate(test_loader, target_scaler)
        
        print("Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Plot predictions
        trainer.plot_predictions(actuals, predictions, n=100)
    
    # Deploy model
    if args.deploy:
        print("Starting API server...")
        from src.deployment.api import app
        import uvicorn
        
        # Save model info for deployment
        model_info = {
            'model_type': args.model,
            'input_size': input_size,
            'sequence_length': args.sequence_length,
            'feature_cols': feature_cols
        }
        
        import json
        with open(os.path.join(args.save_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f)
        
        # Save scalers
        import joblib
        joblib.dump(feature_scaler, os.path.join(args.save_dir, 'feature_scaler.pkl'))
        joblib.dump(target_scaler, os.path.join(args.save_dir, 'target_scaler.pkl'))
        
        # Run API server
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()