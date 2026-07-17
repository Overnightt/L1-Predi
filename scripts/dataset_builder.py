import torch
import pandas as pd
from sample_builder import build_sample
from get_squad_stats import get_squad_stats
from data_check import check_data

#Builds the full dataset from the matches DataFrame.For each match, it constructs the input tensor (TX,T_static_features) 
#and target (y) using `build_sample`.Skips matches that don't have enough prior history.
def build_dataset(matches: pd.DataFrame, n: int, stats: dict):
    TX_list = []
    T_static_features_list = []
    y_list  = []
    squad_values_age = check_data("../data/teams-19-26.csv", date_column="SeasonStart")
    for i in range(len(matches)):
        sample = build_sample(matches,i,n,squad_values_age,stats)
        if sample is not None:
            TX ,T_static_features , y = sample
            TX_list.append(TX)
            T_static_features_list.append(T_static_features)
            y_list.append(y)
    X = torch.stack(TX_list)
    H = torch.stack(T_static_features_list)
    Y = torch.stack(y_list)
    return X, H, Y