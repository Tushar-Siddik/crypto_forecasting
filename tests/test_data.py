# tests/test_data.py
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from datetime import datetime, timedelta

from pathlib import Path

project_root = Path.cwd().parent
sys.path.append(str(project_root))

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import CryptoDataLoader
from data.feature_engineering import FeatureEngineer
from data.preprocessor import DataPreprocessor

class TestDataLoader(unittest.TestCase):
    """Test cases for CryptoDataLoader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = CryptoDataLoader(self.temp_dir)
        
        # Create sample data
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(30000, 60000, len(dates)),
            'High': np.random.uniform(30000, 60000, len(dates)),
            'Low': np.random.uniform(30000, 60000, len(dates)),
            'Close': np.random.uniform(30000, 60000, len(dates)),
            'Volume': np.random.uniform(1000000, 10000000, len(dates))
        }, index=dates)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_fetch_crypto_data(self):
        """Test fetching cryptocurrency data"""
        # Mock the yfinance download
        with unittest.mock.patch('yfinance.download') as mock_download:
            mock_download.return_value = self.sample_data
            
            data = self.loader.fetch_crypto_data('BTC-USD', '2022-01-01', '2022-12-31', save=False)
            
            self.assertIsInstance(data, pd.DataFrame)
            self.assertEqual(len(data), len(self.sample_data))
            self.assertIn('Close', data.columns)
    
    def test_save_and_load_data(self):
        """Test saving and loading data"""
        # Save data
        file_path = os.path.join(self.temp_dir, 'test_data.csv')
        self.sample_data.to_csv(file_path)
        
        # Load data
        loaded_data = self.loader.load_saved_data('BTC-USD', '2022-01-01', '2022-12-31')
        
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(len(loaded_data), len(self.sample_data))

class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engineer = FeatureEngineer()
        
        # Create sample data
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(30000, 60000, len(dates)),
            'High': np.random.uniform(30000, 60000, len(dates)),
            'Low': np.random.uniform(30000, 60000, len(dates)),
            'Close': np.random.uniform(30000, 60000, len(dates)),
            'Volume': np.random.uniform(1000000, 10000000, len(dates))
        }, index=dates)
    
    def test_add_technical_indicators(self):
        """Test adding technical indicators"""
        data_with_features = self.engineer.add_technical_indicators(self.sample_data)
        
        # Check that features were added
        self.assertGreater(len(data_with_features.columns), len(self.sample_data.columns))
        
        # Check for specific indicators
        self.assertIn('RSI', data_with_features.columns)
        self.assertIn('MACD', data_with_features.columns)
        self.assertIn('BB_Upper', data_with_features.columns)
        
        # Check that feature columns were set
        self.assertIsNotNone(self.engineer.feature_columns)
        self.assertGreater(len(self.engineer.feature_columns), 0)
    
    def test_create_sequences(self):
        """Test creating sequences"""
        # Add features first
        data_with_features = self.engineer.add_technical_indicators(self.sample_data)
        
        # Create sequences
        X, y = self.engineer.create_sequences(data_with_features, sequence_length=10)
        
        # Check shapes
        self.assertEqual(X.shape[1], 10)  # sequence length
        self.assertEqual(X.shape[2], len(self.engineer.feature_columns))  # number of features
        self.assertEqual(len(y), X.shape[0])  # same number of samples
    
    def test_select_features(self):
        """Test feature selection"""
        # Add features first
        data_with_features = self.engineer.add_technical_indicators(self.sample_data)
        
        # Select features
        selected = self.engineer.select_features(data_with_features, method='correlation', top_k=10)
        
        # Check selection
        self.assertEqual(len(selected), 10)
        self.assertIn('Close', selected)  # Close should not be in selected features

class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = DataPreprocessor()
        
        # Create sample data
        np.random.seed(42)
        self.X = np.random.randn(100, 10, 5)  # 100 samples, 10 timesteps, 5 features
        self.y = np.random.randn(100)  # 100 targets
    
    def test_split_data(self):
        """Test data splitting"""
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(
            self.X, self.y, train_ratio=0.7, val_ratio=0.15
        )
        
        # Check split ratios
        total_samples = len(self.X)
        self.assertEqual(len(X_train), int(total_samples * 0.7))
        self.assertEqual(len(X_val), int(total_samples * 0.15))
        self.assertEqual(len(X_test), total_samples - len(X_train) - len(X_val))
    
    def test_scale_data(self):
        """Test data scaling"""
        # Split data first
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(
            self.X, self.y, train_ratio=0.7, val_ratio=0.15
        )
        
        # Scale data
        (X_train_scaled, X_val_scaled, X_test_scaled,
         y_train_scaled, y_val_scaled, y_test_scaled) = self.preprocessor.scale_data(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Check that data was scaled
        self.assertLessEqual(X_train_scaled.max(), 1.0)
        self.assertGreaterEqual(X_train_scaled.min(), 0.0)
        self.assertLessEqual(y_train_scaled.max(), 1.0)
        self.assertGreaterEqual(y_train_scaled.min(), 0.0)
    
    def test_inverse_transform_target(self):
        """Test inverse transform of target"""
        # Split and scale data
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(
            self.X, self.y, train_ratio=0.7, val_ratio=0.15
        )
        
        (X_train_scaled, X_val_scaled, X_test_scaled,
         y_train_scaled, y_val_scaled, y_test_scaled) = self.preprocessor.scale_data(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Inverse transform
        y_original = self.preprocessor.inverse_transform_target(y_train_scaled)
        
        # Check that inverse transform works
        self.assertAlmostEqual(y_original[0], y_train[0], places=5)

if __name__ == '__main__':
    unittest.main()