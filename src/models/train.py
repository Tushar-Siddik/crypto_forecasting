# src/models/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import time
from ..data.data_loader import CryptoDataLoader
from ..data.feature_engineering import FeatureEngineer
from .lstm import LSTMModel
from .transformer import TransformerModel
from ..evaluation.metrics import evaluate_model

class TimeSeriesTrainer:
    def __init__(self, model_type='lstm', device=None):
        self.model_type = model_type.lower()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.history = {'train_loss': [], 'val_loss': []}
        
    def prepare_data(self, ticker, sequence_length=60, test_size=0.2, val_size=0.2, random_state=42):
        """
        Prepare data for training
        
        Parameters:
        - ticker: str, cryptocurrency ticker
        - sequence_length: int, length of input sequences
        - test_size: float, proportion of data for testing
        - val_size: float, proportion of training data for validation
        - random_state: int, random seed
        
        Returns:
        - train_loader: DataLoader for training data
        - val_loader: DataLoader for validation data
        - test_loader: DataLoader for test data
        - feature_scaler: fitted scaler for features
        - target_scaler: fitted scaler for target
        - feature_cols: list of feature column names
        """
        # Load data
        loader = CryptoDataLoader()
        data = loader.get_latest_data(ticker, days=730)  # Get 2 years of data
        
        if data is None or len(data) == 0:
            raise ValueError(f"No data found for {ticker}")
        
        # Feature engineering
        engineer = FeatureEngineer()
        data_with_features = engineer.add_technical_indicators(data)
        
        # Define feature columns (exclude target column)
        feature_cols = [col for col in data_with_features.columns if col != 'Close']
        
        # Normalize data
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        # Normalize features
        normalized_features = feature_scaler.fit_transform(data_with_features[feature_cols])
        
        # Normalize target
        normalized_target = target_scaler.fit_transform(data_with_features[['Close']])
        
        # Combine normalized data
        normalized_data = np.concatenate([normalized_features, normalized_target], axis=1)
        
        # Create sequences
        X, y = engineer.create_sequences(
            pd.DataFrame(normalized_data, columns=feature_cols + ['Close'], index=data_with_features.index),
            sequence_length=sequence_length,
            target_col='Close',
            feature_cols=feature_cols
        )
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, shuffle=False)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, feature_scaler, target_scaler, feature_cols
    
    def build_model(self, input_size, output_size=1, **kwargs):
        """
        Build the model based on model_type
        
        Parameters:
        - input_size: int, number of input features
        - output_size: int, number of output values
        - kwargs: additional model-specific parameters
        
        Returns:
        - model: nn.Module
        """
        if self.model_type == 'lstm':
            model = LSTMModel(
                input_size=input_size,
                hidden_size=kwargs.get('hidden_size', 64),
                num_layers=kwargs.get('num_layers', 2),
                output_size=output_size,
                dropout=kwargs.get('dropout', 0.2),
                bidirectional=kwargs.get('bidirectional', False),
                use_gru=kwargs.get('use_gru', False)
            )
        elif self.model_type == 'transformer':
            model = TransformerModel(
                input_size=input_size,
                d_model=kwargs.get('d_model', 64),
                nhead=kwargs.get('nhead', 4),
                num_encoder_layers=kwargs.get('num_encoder_layers', 2),
                dim_feedforward=kwargs.get('dim_feedforward', 128),
                output_size=output_size,
                dropout=kwargs.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model = model.to(self.device)
        return self.model
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=10, save_path=None):
        """
        Train the model
        
        Parameters:
        - train_loader: DataLoader for training data
        - val_loader: DataLoader for validation data
        - epochs: int, number of training epochs
        - lr: float, learning rate
        - patience: int, early stopping patience
        - save_path: str, path to save the best model
        
        Returns:
        - history: dict, training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Early stopping
        best_val_loss = float('inf')
        counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss /= len(val_loader.dataset)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                
                # Save the best model
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load the best model
        if save_path and os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path))
        
        return self.history
    
    def evaluate(self, test_loader, target_scaler=None):
        """
        Evaluate the model on test data
        
        Parameters:
        - test_loader: DataLoader for test data
        - target_scaler: fitted scaler for target values
            
        Returns:
        - metrics: dict, evaluation metrics
        - predictions: numpy array, model predictions
        - actuals: numpy array, actual values
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                
                if target_scaler:
                    # Inverse transform to get actual values
                    outputs_np = outputs.cpu().numpy()
                    targets_np = targets.cpu().numpy()
                    
                    # Reshape for inverse transform
                    outputs_reshaped = outputs_np.reshape(-1, 1)
                    targets_reshaped = targets_np.reshape(-1, 1)
                    
                    # Inverse transform
                    outputs_original = target_scaler.inverse_transform(outputs_reshaped)
                    targets_original = target_scaler.inverse_transform(targets_reshaped)
                    
                    predictions.extend(outputs_original.flatten())
                    actuals.extend(targets_original.flatten())
                else:
                    predictions.extend(outputs.cpu().numpy().flatten())
                    actuals.extend(targets.cpu().numpy().flatten())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        metrics = evaluate_model(actuals, predictions)
        
        return metrics, predictions, actuals
    
    def plot_history(self):
        """Plot training history"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_predictions(self, actuals, predictions, n=100):
        """
        Plot actual vs predicted values
        
        Parameters:
        - actuals: numpy array, actual values
        - predictions: numpy array, predicted values
        - n: int, number of points to plot
        """
        plt.figure(figsize=(12, 6))
        plt.plot(actuals[:n], label='Actual')
        plt.plot(predictions[:n], label='Predicted')
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()