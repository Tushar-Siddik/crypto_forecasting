# data/preprocessor.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Tuple, Optional, Dict, Any
import joblib
import os
import sys
from pathlib import Path

project_root = Path.cwd().parent
sys.path.append(str(project_root))

import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing for time series data"""
    
    def __init__(self, scaler_type: str = 'minmax'):
        self.scaler_type = scaler_type
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_columns = None
        self.target_column = 'Close'
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
        """
        Split data in a time-aware manner
        
        Args:
            X: Input sequences
            y: Target values
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        logger.info(f"Split data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_data(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                   y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Tuple:
        """
        Scale data to improve training stability
        
        Args:
            X_train, X_val, X_test: Input sequences
            y_train, y_val, y_test: Target values
            
        Returns:
            Tuple of scaled data and scalers
        """
        # Initialize scalers
        if self.scaler_type == 'minmax':
            self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.scaler_type == 'standard':
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.feature_scaler = RobustScaler()
            self.target_scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        # Reshape for scaling
        original_X_train_shape = X_train.shape
        original_X_val_shape = X_val.shape
        original_X_test_shape = X_test.shape
        
        # Flatten features for scaling
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        
        # Fit scaler on training data only
        self.feature_scaler.fit(X_train_flat)
        
        # Transform all data
        X_train_scaled = self.feature_scaler.transform(X_train_flat).reshape(original_X_train_shape)
        X_val_scaled = self.feature_scaler.transform(X_val_flat).reshape(original_X_val_shape)
        X_test_scaled = self.feature_scaler.transform(X_test_flat).reshape(original_X_test_shape)
        
        # Scale target separately
        y_train_reshaped = y_train.reshape(-1, 1)
        y_val_reshaped = y_val.reshape(-1, 1)
        y_test_reshaped = y_test.reshape(-1, 1)
        
        self.target_scaler.fit(y_train_reshaped)
        
        y_train_scaled = self.target_scaler.transform(y_train_reshaped).flatten()
        y_val_scaled = self.target_scaler.transform(y_val_reshaped).flatten()
        y_test_scaled = self.target_scaler.transform(y_test_reshaped).flatten()
        
        logger.info(f"Scaled data using {self.scaler_type} scaler")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train_scaled, y_val_scaled, y_test_scaled)
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform target values
        
        Args:
            y: Scaled target values
            
        Returns:
            Original scale target values
        """
        if self.target_scaler is None:
            raise ValueError("Target scaler not fitted")
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        return self.target_scaler.inverse_transform(y).flatten()
    
    def save_scalers(self, save_dir: str) -> None:
        """
        Save fitted scalers to disk
        
        Args:
            save_dir: Directory to save scalers
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if self.feature_scaler is not None:
            joblib.dump(self.feature_scaler, os.path.join(save_dir, 'feature_scaler.pkl'))
        
        if self.target_scaler is not None:
            joblib.dump(self.target_scaler, os.path.join(save_dir, 'target_scaler.pkl'))
        
        if self.feature_columns is not None:
            joblib.dump(self.feature_columns, os.path.join(save_dir, 'feature_columns.pkl'))
        
        logger.info(f"Saved scalers to {save_dir}")
    
    def load_scalers(self, save_dir: str) -> None:
        """
        Load scalers from disk
        
        Args:
            save_dir: Directory to load scalers from
        """
        feature_scaler_path = os.path.join(save_dir, 'feature_scaler.pkl')
        target_scaler_path = os.path.join(save_dir, 'target_scaler.pkl')
        feature_columns_path = os.path.join(save_dir, 'feature_columns.pkl')
        
        if os.path.exists(feature_scaler_path):
            self.feature_scaler = joblib.load(feature_scaler_path)
        
        if os.path.exists(target_scaler_path):
            self.target_scaler = joblib.load(target_scaler_path)
        
        if os.path.exists(feature_columns_path):
            self.feature_columns = joblib.load(feature_columns_path)
        
        logger.info(f"Loaded scalers from {save_dir}")