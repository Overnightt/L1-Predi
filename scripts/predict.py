import torch
import torch.nn as nn
import pandas as pd
from model import FC_Predictor
from data_check import check_data
from sample_builder import build_prediction_sample

#This is where you do your prediction

Model = "../Model/YOUR_MODEL_NAME.pth"
N_MATCHES  = 26
N_FEATURES = 40

HOME_TEAM = input("Home Team:")
AWAY_TEAM = input("Away Team:")

matches = check_data("../data/seasons-19-26.csv")
model = FC_Predictor(N_FEATURES)
model.load_state_dict(torch.load(Model , weights_only=True))
model.eval()
X = build_prediction_sample(matches, HOME_TEAM, AWAY_TEAM, N_MATCHES)
X = X.unsqueeze(0)

with torch.no_grad():
    logits = model(X)                  
    probs = torch.softmax(logits, dim=1)
probs = probs.squeeze().tolist()
print(f"Prediction for {HOME_TEAM} vs {AWAY_TEAM}")
print(f"Home win: {probs[0]:.2%}")
print(f"Draw    : {probs[1]:.2%}")
print(f"Away win: {probs[2]:.2%}")