# training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import time
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..evaluation.metrics import CryptoModelEvaluator
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelTrainer:
    """Enhanced model trainer with advanced features"""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        self.evaluator = CryptoModelEvaluator()
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, lr: float = 0.001, weight_decay: float = 1e-5,
              patience: int = 15, min_delta: float = 1e-6, 
              save_path: Optional[str] = None, scheduler_type: str = 'reduce_on_plateau',
              gradient_clipping: float = 1.0, early_stopping: bool = True) -> Dict[str, List]:
        """
        Train the model with advanced features
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            min_delta: Minimum change to qualify as improvement
            save_path: Path to save the best model
            scheduler_type: Learning rate scheduler type
            gradient_clipping: Gradient clipping value
            early_stopping: Whether to use early stopping
            
        Returns:
            Training history
        """
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        if scheduler_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
            )
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
        else:
            scheduler = None
        
        logger.info(f"Starting training with {epochs} epochs on {self.device}")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = self._train_epoch(train_loader, criterion, optimizer, gradient_clipping)
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader, criterion)
            
            # Update learning rate
            if scheduler:
                if scheduler_type == 'reduce_on_plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Record history
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)
            
            # Print progress
            logger.info(f'Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - '
                       f'train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f} - lr: {current_lr:.2e}')
            
            # Check for improvement
            if val_loss < self.best_val_loss - min_delta:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save the best model
                if save_path:
                    self.model.save_model(save_path)
                    logger.info(f"Model saved to {save_path}")
            else:
                self.patience_counter += 1
                
                if early_stopping and self.patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load the best model
        if save_path and os.path.exists(save_path):
            self.model.load_model(save_path)
        
        logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch+1}")
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader, criterion: nn.Module, 
                    optimizer: optim.Optimizer, gradient_clipping: float) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc="Training", leave=False) as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clipping)
                
                optimizer.step()
                
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation", leave=False) as pbar:
                for inputs, targets in pbar:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets.unsqueeze(1))
                    
                    total_loss += loss.item()
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def evaluate(self, test_loader: DataLoader, target_scaler: Any = None) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluate the model on test data
        
        Args:
            test_loader: Test data loader
            target_scaler: Target scaler for inverse transformation
            
        Returns:
            Tuple of (metrics, predictions, actuals)
        """
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            with tqdm(test_loader, desc="Evaluating") as pbar:
                for inputs, targets in pbar:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    
                    predictions.extend(outputs.cpu().numpy().flatten())
                    actuals.extend(targets.cpu().numpy().flatten())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        self.evaluator.y_scaler = target_scaler
        metrics = self.evaluator.evaluate_model(actuals, predictions)
        
        logger.info("Evaluation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics, predictions, actuals
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Training Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate plot
        axes[0, 1].plot(self.history['learning_rate'])
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True)
        
        # Epoch time plot
        axes[1, 0].plot(self.history['epoch_time'])
        axes[1, 0].set_title('Epoch Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True)
        
        # Loss difference plot
        loss_diff = np.array(self.history['train_loss']) - np.array(self.history['val_loss'])
        axes[1, 1].plot(loss_diff)
        axes[1, 1].set_title('Training - Validation Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions(self, actuals: np.ndarray, predictions: np.ndarray, 
                        n: int = 100, save_path: Optional[str] = None) -> None:
        """Plot predictions vs actual values"""
        self.evaluator.plot_predictions(actuals, predictions, n, save_path=save_path)
    
    def plot_residuals(self, actuals: np.ndarray, predictions: np.ndarray, 
                      save_path: Optional[str] = None) -> None:
        """Plot residuals to analyze model performance"""
        self.evaluator.plot_residuals(actuals, predictions, save_path=save_path)