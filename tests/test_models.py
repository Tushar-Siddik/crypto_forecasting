# tests/test_models.py
import unittest
import torch
import numpy as np
import sys
import os

from pathlib import Path

project_root = Path.cwd().parent
sys.path.append(str(project_root))

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.lstm_attention import LSTMAttention, GRUAttention, MultiHeadAttention
from models.transformer import TransformerModel, InformerModel
from models.ensemble import EnsembleModel

class TestLSTMAttention(unittest.TestCase):
    """Test cases for LSTMAttention model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = LSTMAttention(
            input_size=10,
            hidden_size=32,
            num_layers=2,
            output_size=1,
            dropout=0.1,
            bidirectional=True,
            attention_heads=4
        )
        
        # Create sample input
        self.batch_size = 4
        self.sequence_length = 20
        self.input_size = 10
        self.sample_input = torch.randn(self.batch_size, self.sequence_length, self.input_size)
    
    def test_model_forward(self):
        """Test forward pass"""
        output = self.model(self.sample_input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
    
    def test_model_info(self):
        """Test model information"""
        info = self.model.get_model_info()
        
        # Check info keys
        self.assertIn('model_name', info)
        self.assertIn('total_parameters', info)
        self.assertIn('trainable_parameters', info)
        
        # Check model name
        self.assertEqual(info['model_name'], 'LSTMAttention')
    
    def test_save_load_model(self):
        """Test saving and loading model"""
        import tempfile
        
        with tempfile.NamedTemporaryFile() as tmp:
            # Save model
            self.model.save_model(tmp.name)
            
            # Create new model and load
            new_model = LSTMAttention(
                input_size=10,
                hidden_size=32,
                num_layers=2,
                output_size=1,
                dropout=0.1,
                bidirectional=True,
                attention_heads=4
            )
            new_model.load_model(tmp.name)
            
            # Check that parameters are the same
            for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

class TestGRUAttention(unittest.TestCase):
    """Test cases for GRUAttention model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = GRUAttention(
            input_size=10,
            hidden_size=32,
            num_layers=2,
            output_size=1,
            dropout=0.1,
            bidirectional=True,
            attention_heads=4
        )
        
        # Create sample input
        self.batch_size = 4
        self.sequence_length = 20
        self.input_size = 10
        self.sample_input = torch.randn(self.batch_size, self.sequence_length, self.input_size)
    
    def test_model_forward(self):
        """Test forward pass"""
        output = self.model(self.sample_input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))

class TestTransformerModel(unittest.TestCase):
    """Test cases for TransformerModel"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = TransformerModel(
            input_size=10,
            d_model=32,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=64,
            output_size=1,
            dropout=0.1
        )
        
        # Create sample input
        self.batch_size = 4
        self.sequence_length = 20
        self.input_size = 10
        self.sample_input = torch.randn(self.batch_size, self.sequence_length, self.input_size)
    
    def test_model_forward(self):
        """Test forward pass"""
        output = self.model(self.sample_input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
    
    def test_model_with_mask(self):
        """Test forward pass with mask"""
        # Create mask
        mask = torch.zeros(self.batch_size, self.sequence_length, dtype=torch.bool)
        mask[:, -5:] = True  # Mask last 5 positions
        
        output = self.model(self.sample_input, mask)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))

class TestInformerModel(unittest.TestCase):
    """Test cases for InformerModel"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = InformerModel(
            input_size=10,
            d_model=32,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=1,
            dim_feedforward=64,
            output_size=1,
            dropout=0.1
        )
        
        # Create sample input
        self.batch_size = 4
        self.sequence_length = 20
        self.input_size = 10
        self.sample_input = torch.randn(self.batch_size, self.sequence_length, self.input_size)
    
    def test_model_forward(self):
        """Test forward pass"""
        output = self.model(self.sample_input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))

class TestEnsembleModel(unittest.TestCase):
    """Test cases for EnsembleModel"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create base models
        self.model1 = LSTMAttention(
            input_size=10,
            hidden_size=32,
            num_layers=2,
            output_size=1,
            dropout=0.1
        )
        
        self.model2 = TransformerModel(
            input_size=10,
            d_model=32,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=64,
            output_size=1,
            dropout=0.1
        )
        
        # Create ensemble
        self.ensemble = EnsembleModel(
            models=[self.model1, self.model2],
            input_size=10,
            output_size=1,
            aggregation_method='average'
        )
        
        # Create sample input
        self.batch_size = 4
        self.sequence_length = 20
        self.input_size = 10
        self.sample_input = torch.randn(self.batch_size, self.sequence_length, self.input_size)
    
    def test_ensemble_forward(self):
        """Test forward pass"""
        output = self.ensemble(self.sample_input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
    
    def test_weighted_average_ensemble(self):
        """Test weighted average aggregation"""
        ensemble = EnsembleModel(
            models=[self.model1, self.model2],
            input_size=10,
            output_size=1,
            aggregation_method='weighted_average'
        )
        
        output = ensemble(self.sample_input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
    
    def test_attention_ensemble(self):
        """Test attention aggregation"""
        ensemble = EnsembleModel(
            models=[self.model1, self.model2],
            input_size=10,
            output_size=1,
            aggregation_method='attention'
        )
        
        output = ensemble(self.sample_input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
    
    def test_stacking_ensemble(self):
        """Test stacking aggregation"""
        ensemble = EnsembleModel(
            models=[self.model1, self.model2],
            input_size=10,
            output_size=1,
            aggregation_method='stacking'
        )
        
        output = ensemble(self.sample_input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))

class TestMultiHeadAttention(unittest.TestCase):
    """Test cases for MultiHeadAttention"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.attention = MultiHeadAttention(
            hidden_size=32,
            num_heads=4,
            dropout=0.1
        )
        
        # Create sample input
        self.batch_size = 4
        self.sequence_length = 20
        self.hidden_size = 32
        self.sample_input = torch.randn(self.batch_size, self.sequence_length, self.hidden_size)
    
    def test_attention_forward(self):
        """Test forward pass"""
        output = self.attention(self.sample_input)
        
        # Check output shape
        self.assertEqual(output.shape, self.sample_input.shape)
    
    def test_attention_with_mask(self):
        """Test forward pass with mask"""
        # Create mask
        mask = torch.zeros(self.batch_size, self.sequence_length, dtype=torch.bool)
        mask[:, -5:] = True  # Mask last 5 positions
        
        output = self.attention(self.sample_input, mask)
        
        # Check output shape
        self.assertEqual(output.shape, self.sample_input.shape)

if __name__ == '__main__':
    unittest.main()