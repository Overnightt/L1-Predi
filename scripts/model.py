import torch
import torch.nn as nn

# L1_Predictor is a simple fully connected neural network for predicting the outcome of a Ligue 1 match.
# Input: a tensor of shape [batch_size, n_matches, n_features] (past match sequences for home & away teams)
# Output: a tensor of shape [batch_size, 3] with raw scores for each class (Home win, Draw, Away win)
class L1_Predictor(nn.Module):
    def __init__(self,n_features: int,hidden_size: int = 16, num_layers = 2):
        super().__init__()
        self.lstm= nn.LSTM(
            input_size = n_features,
            hidden_size = hidden_size,
            num_layers = num_layers
            batch_first = True   
        )
        self.fc = nn.Linear(hidden_size,3)
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.fully_connected_layer1(x)
        x = self.relu(x)
        x= self.fully_connected_layer2(x)
        return x