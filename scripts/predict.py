import torch
import torch.nn as nn
import pandas as pd
from model import FC_Predictor
from data_check import check_data
from sample_builder import build_prediction_sample
from normalisation_stats import norm_stats

# This is where you do your prediction

Model = "../Model/YOUR_MODEL_NAME.pth"
N_MATCHES  = 26
N_FEATURES = 40
STATIC_SIZE = 5

HOME_TEAM = input("Home Team:")
AWAY_TEAM = input("Away Team:")

matches = check_data("../data/seasons-19-26.csv")
squad_values = check_data("../data/teams-19-26.csv", date_column="SeasonStart")

# No train/val split at prediction time, so we compute norm_stats on all available matches
stats = norm_stats(matches, squad_values)

model = FC_Predictor(N_FEATURES, STATIC_SIZE)
model.load_state_dict(torch.load(Model, weights_only=True))
model.eval()

sample = build_prediction_sample(matches, HOME_TEAM, AWAY_TEAM, N_MATCHES, squad_values, stats)
if sample is None:
    print("Not enough historical data or squad data to make a prediction for these teams.")
else:
    X, static = sample
    X = X.unsqueeze(0)
    static = static.unsqueeze(0)

    with torch.no_grad():
        logits = model(X, static)
        probs = torch.softmax(logits, dim=1)
    probs = probs.squeeze().tolist()
    print(f"Prediction for {HOME_TEAM} vs {AWAY_TEAM}")
    print(f"Home win: {probs[0]:.2%}")
    print(f"Draw    : {probs[1]:.2%}")
    print(f"Away win: {probs[2]:.2%}")