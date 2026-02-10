import torch
import torch.nn as nn
import pandas as pd
from model import L1_Predictor
from data_check import check_data
from sample_builder import build_prediction_sample

MODEL_PATH = "../Model/L1_predictor_v8_26.pth"
DATA_PATH  = "../data/seasons-17-26.csv"  
N_MATCHES  = 26
N_FEATURES = 40

HOME_TEAM = input("Home Team:")
AWAY_TEAM = input("Away Team:")

matches = check_data(DATA_PATH)
model = L1_Predictor(N_MATCHES, N_FEATURES)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
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