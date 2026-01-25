import torch
import pandas as pd
from data_check import check_data
from build_sequences import build_sequence


#A list of all teams
all_team_names = ['Le Havre', 'Paris SG', 'Brest', 'Marseille', 'Reims', 'Lille', 'Monaco', 'St Etienne', 'Auxerre', 'Nice', 'Angers', 'Lens', 'Montpellier', 'Strasbourg', 'Toulouse', 'Rennes', 'Lyon','Nantes']

# Converts a football match sequence DataFrame into a PyTorch tensor suitable for ML models:
#   -Drops the 'Date' and 'Referee" column as they are not used in modeling.
#   -Encodes categorical outcome columns ('FTR' and 'HTR') into numeric codes.
#   -Encodes team names ('HomeTeam' and 'AwayTeam') into numeric codes based on all_team_names.
#   -Converts the entire DataFrame into a PyTorch tensor of type float
def sequence_to_tensor(sequence: pd.DataFrame):
    sequence = sequence.drop(columns=["Date","Referee"])
    sequence["FTR"] = pd.Categorical(sequence["FTR"], categories=["H","D","A"]).codes
    sequence["HTR"] = pd.Categorical(sequence["HTR"], categories=["H","D","A"]).codes
    sequence["HomeTeam"] = pd.Categorical(sequence["HomeTeam"], categories=all_team_names).codes
    sequence["AwayTeam"] = pd.Categorical(sequence["AwayTeam"], categories=all_team_names).codes
    return torch.tensor(sequence.values,dtype=torch.float)

