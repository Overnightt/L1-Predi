import torch
import torch.nn as nn

# L1_Predictor is a simple fully connected neural network for predicting the outcome of a Ligue 1 match.
# Input: a tensor of shape [batch_size, n_matches, n_features] (past match sequences for home & away teams)
# Output: a tensor of shape [batch_size, 3] with raw scores for each class (Home win, Draw, Away win)
class L1_Predictor(nn.Module):
    def __init__(self,n_matches: int,n_features: int):
        super().__init__()
        input_size = n_matches*n_features
        self.fully_connected_layer1 = nn.Linear(input_size,32)
        self.relu = nn.ReLU()
        self.fully_connected_layer2 = nn.Linear(32,3)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.fully_connected_layer1(x)
        x = self.relu(x)
        x= self.fully_connected_layer2(x)
        return x