# src/models/lstm.py
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, bidirectional=False, use_gru=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.direction_factor = 2 if bidirectional else 1
        
        # Choose the appropriate RNN layer
        if use_gru:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, 
                             dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                              dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.direction_factor, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * self.direction_factor, x.size(0), self.hidden_size).to(x.device)
        
        # Initialize cell state for LSTM only
        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros(self.num_layers * self.direction_factor, x.size(0), self.hidden_size).to(x.device)
            # Forward propagate LSTM
            out, _ = self.rnn(x, (h0, c0))
        else:
            # Forward propagate GRU
            out, _ = self.rnn(x, h0)
        
        # Get the output of the last time step
        if self.bidirectional:
            # Concatenate the final forward and backward hidden states
            out = torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), dim=1)
        else:
            out = out[:, -1, :]
        
        # Apply dropout and fully connected layer
        out = self.dropout(out)
        out = self.fc(out)
        
        return out