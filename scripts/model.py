import torch
import torch.nn as nn

# FC_Predictor is a Long Short-Term Memory (LSTM) network for predicting the outcome of a football match,
# combined with a set of static (non-sequential) match features processed by a final linear layer.
# Inputs:
#   x      - tensor of shape [batch_size, n_matches, n_features] (past match sequences for home & away teams)
#   static - tensor of shape [batch_size, static_size] (static features like head-to-head score)
# Output: a tensor of shape [batch_size, 3] with raw scores for each class (Home win, Draw, Away win)
class FC_Predictor(nn.Module):
    def __init__(self, n_features: int, static_size: int, hidden_size: int = 16, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size + static_size, 3)

    def forward(self, x, static):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        combined = torch.cat((last_out, static), dim=1)
        return self.fc(combined)