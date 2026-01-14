# tests/test_integration.py
import unittest
import torch
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import CryptoDataLoader
from data.feature_engineering import FeatureEngineer
from data.preprocessor import DataPreprocessor
from models.lstm_attention import LSTMAttention
from training.trainer import ModelTrainer
from evaluation.metrics import CryptoModelEvaluator
from torch.utils.data import DataLoader, TensorDataset

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
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
    
    def test_complete_pipeline(self):
        """Test the complete pipeline from data to evaluation"""
        # 1. Data loading
        loader = CryptoDataLoader(self.temp_dir)
        
        # Save sample data
        data_path = os.path.join(self.temp_dir, 'test_data.csv')
        self.sample_data.to_csv(data_path)
        
        # Load data
        data = loader.load_saved_data('TEST-USD', '2022-01-01', '2022-12-31')
        self.assertIsNotNone(data)
        self.assertEqual(len(data), len(self.sample_data))
        
        # 2. Feature engineering
        engineer = FeatureEngineer()
        data_with_features = engineer.add_technical_indicators(data)
        
        # Check features were added
        self.assertGreater(len(data_with_features.columns), len(data.columns))
        self.assertIsNotNone(engineer.feature_columns)
        
        # 3. Create sequences
        X, y = engineer.create_sequences(data_with_features, sequence_length=10)
        
        # Check sequence shapes
        self.assertEqual(X.shape[1], 10)  # sequence length
        self.assertEqual(X.shape[2], len(engineer.feature_columns))  # features
        self.assertEqual(len(y), X.shape[0])  # same number of samples
        
        # 4. Data preprocessing
        preprocessor = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            X, y, train_ratio=0.7, val_ratio=0.15
        )
        
        # Check split
        total_samples = len(X)
        self.assertEqual(len(X_train), int(total_samples * 0.7))
        self.assertEqual(len(X_val), int(total_samples * 0.15))
        self.assertEqual(len(X_test), total_samples - len(X_train) - len(X_val))
        
        # Scale data
        (X_train_scaled, X_val_scaled, X_test_scaled,
         y_train_scaled, y_val_scaled, y_test_scaled) = preprocessor.scale_data(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Check scaling
        self.assertLessEqual(X_train_scaled.max(), 1.0)
        self.assertGreaterEqual(X_train_scaled.min(), 0.0)
        
        # 5. Model training
        model = LSTMAttention(
            input_size=len(engineer.feature_columns),
            hidden_size=16,
            num_layers=1,
            output_size=1,
            dropout=0.1
        )
        
        # Create data loaders
        from torch.utils.data import DataLoader, TensorDataset
        
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
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Train model
        trainer = ModelTrainer(model, device='cpu')
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,  # Reduced for testing
            lr=0.01,
            patience=1,
            early_stopping=True
        )
        
        # Check training history
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertEqual(len(history['train_loss']), 2)
        
        # 6. Evaluation
        metrics, predictions, actuals = trainer.evaluate(
            test_loader, preprocessor.target_scaler
        )
        
        # Check metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        
        # Check predictions
        self.assertEqual(len(predictions), len(actuals))
        self.assertGreater(len(predictions), 0)
        
        # 7. Save components
        model_save_dir = os.path.join(self.temp_dir, 'model')
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Save model
        model.save_model(os.path.join(model_save_dir, 'model.pth'))
        
        # Save scalers
        preprocessor.save_scalers(model_save_dir)
        
        # Check files were saved
        self.assertTrue(os.path.exists(os.path.join(model_save_dir, 'model.pth')))
        self.assertTrue(os.path.exists(os.path.join(model_save_dir, 'feature_scaler.pkl')))
        self.assertTrue(os.path.exists(os.path.join(model_save_dir, 'target_scaler.pkl')))
    
    def test_model_persistence(self):
        """Test model saving and loading"""
        # Create and train a simple model
        model = LSTMAttention(
            input_size=5,
            hidden_size=16,
            num_layers=1,
            output_size=1,
            dropout=0.1
        )
        
        # Save model
        model_path = os.path.join(self.temp_dir, 'test_model.pth')
        model.save_model(model_path)
        
        # Load model
        new_model = LSTMAttention(
            input_size=5,
            hidden_size=16,
            num_layers=1,
            output_size=1,
            dropout=0.1
        )
        new_model.load_model(model_path)
        
        # Check parameters are the same
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))
    
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction pipeline"""
        # Create a simple pipeline
        engineer = FeatureEngineer()
        preprocessor = DataPreprocessor()
        
        # Add features to sample data
        data_with_features = engineer.add_technical_indicators(self.sample_data)
        
        # Create sequences
        X, y = engineer.create_sequences(data_with_features, sequence_length=10)
        
        # Split and scale
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        (X_train_scaled, X_val_scaled, X_test_scaled,
         y_train_scaled, y_val_scaled, y_test_scaled) = preprocessor.scale_data(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Train model
        model = LSTMAttention(
            input_size=len(engineer.feature_columns),
            hidden_size=16,
            num_layers=1,
            output_size=1,
            dropout=0.1
        )
        
        trainer = ModelTrainer(model, device='cpu')
        trainer.train(
            train_loader=DataLoader(
                TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled)),
                batch_size=8, shuffle=True
            ),
            val_loader=DataLoader(
                TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val_scaled)),
                batch_size=8, shuffle=False
            ),
            epochs=1,
            lr=0.01,
            patience=1
        )
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            sample_input = torch.FloatTensor(X_test_scaled[:1])
            prediction = model(sample_input)
        
        # Check prediction
        self.assertEqual(prediction.shape, (1, 1))
        
        # Inverse transform
        original_prediction = preprocessor.inverse_transform_target(
            prediction.numpy().flatten()
        )
        
        # Check inverse transform
        self.assertEqual(len(original_prediction), 1)
        self.assertIsInstance(original_prediction[0], float)

if __name__ == '__main__':
    unittest.main()