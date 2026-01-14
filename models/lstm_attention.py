# models/lstm_attention.py
import torch
import torch.nn as nn
import math
from typing import Optional
from .base_model import BaseModel

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        output = self.out(context)
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer-style models"""
    
    def __init__(self, hidden_size: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class LSTMAttention(BaseModel):
    """LSTM with Attention mechanism"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2, bidirectional: bool = True,
                 attention_heads: int = 8, **kwargs):
        super(LSTMAttention, self).__init__(input_size, output_size, **kwargs)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.direction_factor = 2 if bidirectional else 1
        self.attention_heads = attention_heads
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size * self.direction_factor,
            num_heads=attention_heads,
            dropout=dropout
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * self.direction_factor)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * self.direction_factor, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * self.direction_factor, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.direction_factor, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        attended = self.attention(lstm_out)
        
        # Residual connection and layer normalization
        attended = self.layer_norm(lstm_out + attended)
        
        # Use the output of the last time step
        if self.bidirectional:
            # Concatenate the final forward and backward hidden states
            last_output = torch.cat((attended[:, -1, :self.hidden_size], attended[:, 0, self.hidden_size:]), dim=1)
        else:
            last_output = attended[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out

class GRUAttention(BaseModel):
    """GRU with Attention mechanism"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2, bidirectional: bool = True,
                 attention_heads: int = 8, **kwargs):
        super(GRUAttention, self).__init__(input_size, output_size, **kwargs)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.direction_factor = 2 if bidirectional else 1
        self.attention_heads = attention_heads
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size * self.direction_factor,
            num_heads=attention_heads,
            dropout=dropout
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * self.direction_factor)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * self.direction_factor, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * self.direction_factor, batch_size, self.hidden_size).to(x.device)
        
        # GRU forward pass
        gru_out, _ = self.gru(x, h0)
        
        # Apply attention
        attended = self.attention(gru_out)
        
        # Residual connection and layer normalization
        attended = self.layer_norm(gru_out + attended)
        
        # Use the output of the last time step
        if self.bidirectional:
            # Concatenate the final forward and backward hidden states
            last_output = torch.cat((attended[:, -1, :self.hidden_size], attended[:, 0, self.hidden_size:]), dim=1)
        else:
            last_output = attended[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out