# models/transformer.py
import torch
import torch.nn as nn
import math
from typing import Optional
import sys
from pathlib import Path

project_root = Path.cwd().parent
sys.path.append(str(project_root))

from models.base_model import BaseModel
from models.lstm_attention import PositionalEncoding

class TransformerModel(BaseModel):
    """Transformer model for time series forecasting"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_encoder_layers: int = 3, dim_feedforward: int = 256, 
                 output_size: int = 1, dropout: float = 0.1, **kwargs):
        super(TransformerModel, self).__init__(input_size, output_size, **kwargs)
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, d_model // 4)
        self.fc3 = nn.Linear(d_model // 4, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Use the output of the last time step
        x = x[:, -1, :]
        
        # Apply fully connected layers
        x = self.dropout(x)
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class InformerModel(BaseModel):
    """Informer model for long sequence time series forecasting"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_encoder_layers: int = 2, num_decoder_layers: int = 1,
                 dim_feedforward: int = 256, output_size: int = 1, 
                 dropout: float = 0.1, **kwargs):
        super(InformerModel, self).__init__(input_size, output_size, **kwargs)
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # ProbSparse self-attention (simplified version)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_encoder_layers)
        ])
        
        # Distillation layers (simplified)
        self.distill = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
            for _ in range(num_encoder_layers - 1)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply encoder layers with distillation
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            
            # Apply distillation (except for the last layer)
            if i < len(self.distill):
                # Reshape for Conv1d
                x_conv = x.permute(0, 2, 1)
                x_conv = self.distill[i](x_conv)
                x = x_conv.permute(0, 2, 1)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Use the output of the last time step
        x = x[:, -1, :]
        
        # Apply fully connected layers
        x = self.dropout(x)
        x = self.elu(self.fc1(x))
        x = self.fc2(x)
        
        return x