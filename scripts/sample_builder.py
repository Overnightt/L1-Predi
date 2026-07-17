import torch
import pandas as pd
from build_sequences import build_sequence
from sequence_to_tensor import sequence_to_tensor
from head_to_head import h2h_score
from get_squad_stats import get_squad_stats

#Builds a training sample for a single match. Skips matches that don't have enough prior history.
def build_sample(matches: pd.DataFrame,row : int,n: int, squad_values_age: pd.DataFrame, stats: dict):
    match= matches.iloc[row]
    date= match["Date"]
    HT= match["HomeTeam"]
    AT= match["AwayTeam"]

    seqH = build_sequence(HT,date,matches,n)
    seqA = build_sequence(AT,date,matches,n)
    H2H = h2h_score(HT,AT,date,matches)
    if len(seqH) <n or len(seqA)<n:
        return None
    home_stats = get_squad_stats(HT, date, squad_values_age)
    away_stats = get_squad_stats(AT, date, squad_values_age)
    if home_stats is None or away_stats is None:
        return None
    HTA, HTV = home_stats
    ATA, ATV = away_stats

    TH = sequence_to_tensor(seqH)
    TA = sequence_to_tensor(seqA)
    TX = torch.cat((TH,TA),dim=1)
    HTA_norm = (HTA - stats["age_mean"]) / stats["age_std"]
    HTV_norm = (HTV - stats["value_mean"]) / stats["value_std"]
    ATA_norm = (ATA - stats["age_mean"]) / stats["age_std"]
    ATV_norm = (ATV - stats["value_mean"]) / stats["value_std"]
    T_static_features = torch.tensor([H2H, HTA_norm, HTV_norm, ATA_norm, ATV_norm], dtype=torch.float32)
    outcome_map = {"H":0,"D":1,"A":2}
    y = torch.tensor([outcome_map[match["FTR"]]],dtype=torch.long)
    return TX,T_static_features,y

#Buils a sample but for prediction
def build_prediction_sample(matches: pd.DataFrame, home_team: str, away_team: str, n: int, squad_values_age: pd.DataFrame, stats: dict):
    current_date = matches["Date"].max()
    
    seqH = build_sequence(home_team, current_date, matches, n)
    seqA = build_sequence(away_team, current_date, matches, n)
    H2H = h2h_score(home_team, away_team, current_date, matches)
    if len(seqH) < n or len(seqA) < n:
        return None
    home_stats = get_squad_stats(home_team, current_date, squad_values_age)
    away_stats = get_squad_stats(away_team, current_date, squad_values_age)
    if home_stats is None or away_stats is None:
        return None
    HTA, HTV = home_stats
    ATA, ATV = away_stats

    TH = sequence_to_tensor(seqH)
    TA = sequence_to_tensor(seqA)
    TX = torch.cat((TH, TA), dim=1)
    HTA_norm = (HTA - stats["age_mean"]) / stats["age_std"]
    HTV_norm = (HTV - stats["value_mean"]) / stats["value_std"]
    ATA_norm = (ATA - stats["age_mean"]) / stats["age_std"]
    ATV_norm = (ATV - stats["value_mean"]) / stats["value_std"]
    T_static_features = torch.tensor([H2H, HTA_norm, HTV_norm, ATA_norm, ATV_norm], dtype=torch.float32)
    return TX, T_static_features