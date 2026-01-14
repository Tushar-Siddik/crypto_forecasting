# tests/test_training.py
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import tempfile
import os
import sys

from pathlib import Path

project_root = Path.cwd().parent
sys.path.append(str(project_root))

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.trainer import ModelTrainer
from training.hyperparameter_tuning import HyperparameterTuner
from models.lstm_attention import LSTMAttention

class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create model
        self.model = LSTMAttention(
            input_size=5,
            hidden_size=16,
            num_layers=1,
            output_size=1,
            dropout=0.1
        )
        
        # Create sample data
        self.batch_size = 8
        self.sequence_length = 10
        self.input_size = 5
        
        # Generate sample data
        X = torch.randn(100, self.sequence_length, self.input_size)
        y = torch.randn(100)
        
        # Create datasets and loaders
        train_dataset = TensorDataset(X[:80], y[:80])
        val_dataset = TensorDataset(X[80:90], y[80:90])
        test_dataset = TensorDataset(X[90:], y[90:])
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Create trainer
        self.trainer = ModelTrainer(self.model, device='cpu')
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        self.assertIsNotNone(self.trainer.model)
        self.assertEqual(self.trainer.device.type, 'cpu')
        self.assertIsInstance(self.trainer.history, dict)
    
    def test_train_epoch(self):
        """Test training for one epoch"""
        train_loss = self.trainer._train_epoch(
            self.train_loader, 
            nn.MSELoss(), 
            torch.optim.Adam(self.model.parameters()),
            gradient_clipping=1.0
        )
        
        # Check that loss is a float
        self.assertIsInstance(train_loss, float)
        self.assertGreater(train_loss, 0)
    
    def test_validate_epoch(self):
        """Test validation for one epoch"""
        val_loss = self.trainer._validate_epoch(
            self.val_loader, 
            nn.MSELoss()
        )
        
        # Check that loss is a float
        self.assertIsInstance(val_loss, float)
        self.assertGreater(val_loss, 0)
    
    def test_train_model(self):
        """Test full training process"""
        # Train for 2 epochs
        history = self.trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=2,
            lr=0.01,
            patience=1,
            early_stopping=True
        )
        
        # Check history
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertEqual(len(history['train_loss']), 2)
        self.assertEqual(len(history['val_loss']), 2)
    
    def test_evaluate_model(self):
        """Test model evaluation"""
        # Train for 1 epoch first
        self.trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=1,
            lr=0.01,
            patience=1,
            early_stopping=True
        )
        
        # Evaluate
        metrics, predictions, actuals = self.trainer.evaluate(self.test_loader)
        
        # Check metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        
        # Check predictions and actuals
        self.assertEqual(len(predictions), len(actuals))
        self.assertEqual(len(predictions), 10)  # Test set has 10 samples

class TestHyperparameterTuner(unittest.TestCase):
    """Test cases for HyperparameterTuner"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_type = 'lstm_attention'
        self.input_size = 5
        self.output_size = 1
        
        # Create sample data
        X = torch.randn(100, 10, 5)
        y = torch.randn(100)
        
        # Create datasets and loaders
        train_dataset = TensorDataset(X[:80], y[:80])
        val_dataset = TensorDataset(X[80:90], y[80:90])
        
        self.train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Create tuner
        self.tuner = HyperparameterTuner(
            model_type=self.model_type,
            input_size=self.input_size,
            output_size=self.output_size,
            device='cpu',
            n_trials=2  # Reduced for testing
        )
    
    def test_tuner_initialization(self):
        """Test tuner initialization"""
        self.assertEqual(self.tuner.model_type, self.model_type)
        self.assertEqual(self.tuner.input_size, self.input_size)
        self.assertEqual(self.tuner.output_size, self.output_size)
        self.assertEqual(self.tuner.n_trials, 2)
    
    def test_objective_function(self):
        """Test objective function"""
        import optuna
        
        # Create a trial
        trial = optuna.trial.Trial(optuna.study.create_study())
        
        # Mock trial parameters
        trial.suggest_categorical = lambda name, choices: choices[0]
        trial.suggest_int = lambda name, low, high: low
        trial.suggest_float = lambda name, low, high, log=True: low
        
        # Test objective function
        loss = self.tuner.objective(trial, self.train_loader, self.val_loader)
        
        # Check that loss is returned
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
    
    def test_get_best_model(self):
        """Test getting best model"""
        # Set mock best parameters
        self.tuner.best_params = {
            'hidden_size': 32,
            'num_layers': 2,
            'dropout': 0.1,
            'bidirectional': True,
            'attention_heads': 4
        }
        
        # Get best model
        model = self.tuner.get_best_model()
        
        # Check model
        self.assertIsInstance(model, LSTMAttention)
        self.assertEqual(model.hidden_size, 32)
        self.assertEqual(model.num_layers, 2)

if __name__ == '__main__':
    unittest.main()