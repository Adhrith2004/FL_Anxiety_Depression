# The 1D-CNN + LSTM Multi-Task model

import torch
import torch.nn as nn
from src.config import WINDOW_LEN

class CnnLstmMultiTask(nn.Module):
    def __init__(self, in_channels=3, lstm_hidden=128):
        """
        Model for Multi-Task Learning.
        Input: (batch_size, 3_channels, 1000_samples)
        Output: (pred_anxiety, pred_negaffect)
        """
        super(CnnLstmMultiTask, self).__init__()
        
        # --- 1D-CNN Feature Extractor ---
        # Input: (B, 3, 1000)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3), # (B, 32, 500)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4), # (B, 32, 125)
            
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2), # (B, 64, 125)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5)  # (B, 64, 25)
        )
        
        # --- LSTM Sequence Model ---
        # The CNN outputs 64 channels (features) over 25 time steps.
        # LSTM expects (batch_size, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=lstm_hidden, 
            num_layers=1, 
            batch_first=True
        )
        
        # --- Shared FC Layer ---
        self.fc_shared = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # --- Multi-Task Heads (Regression) ---
        # Head 1: Predicts Anxiety (a continuous score)
        self.head_anxiety = nn.Linear(64, 1)
        
        # Head 2: Predicts Negative Affect (a continuous score)
        self.head_negaffect = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (B, 3, 1000)
        
        x = self.cnn(x) 
        # x shape: (B, 64, 25)
        
        # Reshape for LSTM: (B, 64, 25) -> (B, 25, 64)
        x = x.permute(0, 2, 1) 
        
        # We only care about the last hidden state
        _, (h_n, _) = self.lstm(x)
        x = h_n[0] # Get the last layer's hidden state: (B, 128)
        
        x = self.fc_shared(x) # (B, 64)
        
        # We are predicting scores (regression), so no activation fn.
        pred_anxiety = self.head_anxiety(x)   # (B, 1)
        pred_negaffect = self.head_negaffect(x) # (B, 1)
        
        return pred_anxiety, pred_negaffect