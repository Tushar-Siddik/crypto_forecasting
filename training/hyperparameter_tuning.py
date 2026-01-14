# training/hyperparameter_tuning.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import json
import os
from datetime import datetime
import sys
from pathlib import Path

project_root = Path.cwd().parent
sys.path.append(str(project_root))

from training.trainer import ModelTrainer
from models.lstm_attention import LSTMAttention, GRUAttention
from models.transformer import TransformerModel, InformerModel
from utils.logger import logger
from utils.logger import setup_logger

logger = setup_logger(__name__)

class HyperparameterTuner:
    """Hyperparameter tuning using Optuna"""
    
    def __init__(self, model_type: str, input_size: int, output_size: int = 1,
                 device: Optional[torch.device] = None, n_trials: int = 100,
                 timeout: Optional[int] = None):
        self.model_type = model_type
        self.input_size = input_size
        self.output_size = output_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_trials = n_trials
        self.timeout = timeout
        
        self.best_params = None
        self.best_value = None
        self.study = None
    
    def objective(self, trial: optuna.Trial, train_loader: DataLoader, 
                 val_loader: DataLoader) -> float:
        """Objective function for optimization"""
        # Sample hyperparameters
        if self.model_type == 'lstm_attention':
            hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            bidirectional = trial.suggest_categorical('bidirectional', [True, False])
            attention_heads = trial.suggest_categorical('attention_heads', [4, 8, 16])
            
            model = LSTMAttention(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=self.output_size,
                dropout=dropout,
                bidirectional=bidirectional,
                attention_heads=attention_heads
            )
        
        elif self.model_type == 'gru_attention':
            hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            bidirectional = trial.suggest_categorical('bidirectional', [True, False])
            attention_heads = trial.suggest_categorical('attention_heads', [4, 8, 16])
            
            model = GRUAttention(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=self.output_size,
                dropout=dropout,
                bidirectional=bidirectional,
                attention_heads=attention_heads
            )
        
        elif self.model_type == 'transformer':
            d_model = trial.suggest_categorical('d_model', [64, 128, 256])
            nhead = trial.suggest_categorical('nhead', [4, 8, 16])
            num_encoder_layers = trial.suggest_int('num_encoder_layers', 2, 4)
            dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512])
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            
            model = TransformerModel(
                input_size=self.input_size,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                output_size=self.output_size,
                dropout=dropout
            )
        
        elif self.model_type == 'informer':
            d_model = trial.suggest_categorical('d_model', [64, 128, 256])
            nhead = trial.suggest_categorical('nhead', [4, 8, 16])
            num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 3)
            num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 2)
            dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512])
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            
            model = InformerModel(
                input_size=self.input_size,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                output_size=self.output_size,
                dropout=dropout
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Training hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        
        # Update data loaders with new batch size
        train_loader.dataset.batch_size = batch_size
        val_loader.dataset.batch_size = batch_size
        
        # Create trainer
        trainer = ModelTrainer(model, self.device)
        
        # Train model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=50,  # Use fewer epochs for faster tuning
            lr=lr,
            weight_decay=weight_decay,
            patience=10,
            early_stopping=True
        )
        
        # Return best validation loss
        return trainer.best_val_loss
    
    def tune(self, train_loader: DataLoader, val_loader: DataLoader,
             save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Tune hyperparameters
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_path: Path to save the best hyperparameters
            
        Returns:
            Dictionary of best hyperparameters
        """
        # Create study
        self.study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Optimize
        logger.info(f"Starting hyperparameter tuning for {self.model_type}")
        
        self.study.optimize(
            lambda trial: self.objective(trial, train_loader, val_loader),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        logger.info(f"Best validation loss: {self.best_value:.6f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        # Save best parameters
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(self.best_params, f, indent=4)
            logger.info(f"Best hyperparameters saved to {save_path}")
        
        return self.best_params
    
    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """Plot optimization history"""
        try:
            import optuna.visualization as vis
            
            # Plot optimization history
            fig = vis.plot_optimization_history(self.study)
            fig.show()
            
            if save_path:
                fig.write_html(f"{save_path}_optimization_history.html")
            
            # Plot parameter importance
            fig = vis.plot_param_importances(self.study)
            fig.show()
            
            if save_path:
                fig.write_html(f"{save_path}_param_importance.html")
            
            # Plot parallel coordinate
            fig = vis.plot_parallel_coordinate(self.study)
            fig.show()
            
            if save_path:
                fig.write_html(f"{save_path}_parallel_coordinate.html")
                
        except ImportError:
            logger.warning("Optuna visualization not available. Install optuna[visualization] for plots.")
    
    def get_best_model(self) -> nn.Module:
        """Create a model with the best hyperparameters"""
        if self.best_params is None:
            raise ValueError("No best parameters found. Run tune() first.")
        
        if self.model_type == 'lstm_attention':
            return LSTMAttention(
                input_size=self.input_size,
                hidden_size=self.best_params['hidden_size'],
                num_layers=self.best_params['num_layers'],
                output_size=self.output_size,
                dropout=self.best_params['dropout'],
                bidirectional=self.best_params['bidirectional'],
                attention_heads=self.best_params['attention_heads']
            )
        
        elif self.model_type == 'gru_attention':
            return GRUAttention(
                input_size=self.input_size,
                hidden_size=self.best_params['hidden_size'],
                num_layers=self.best_params['num_layers'],
                output_size=self.output_size,
                dropout=self.best_params['dropout'],
                bidirectional=self.best_params['bidirectional'],
                attention_heads=self.best_params['attention_heads']
            )
        
        elif self.model_type == 'transformer':
            return TransformerModel(
                input_size=self.input_size,
                d_model=self.best_params['d_model'],
                nhead=self.best_params['nhead'],
                num_encoder_layers=self.best_params['num_encoder_layers'],
                dim_feedforward=self.best_params['dim_feedforward'],
                output_size=self.output_size,
                dropout=self.best_params['dropout']
            )
        
        elif self.model_type == 'informer':
            return InformerModel(
                input_size=self.input_size,
                d_model=self.best_params['d_model'],
                nhead=self.best_params['nhead'],
                num_encoder_layers=self.best_params['num_encoder_layers'],
                num_decoder_layers=self.best_params['num_decoder_layers'],
                dim_feedforward=self.best_params['dim_feedforward'],
                output_size=self.output_size,
                dropout=self.best_params['dropout']
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")