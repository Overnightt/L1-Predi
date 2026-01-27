import torch
import pandas as pd
from sample_builder import build_sample

#Builds the full dataset from the matches DataFrame.For each match, it constructs the input tensor (TX) 
#and target (y) using `build_sample`.Skips matches that don't have enough prior history.
def build_dataset(matches: pd.DataFrame, n: int):
    TX_list = []
    y_list  = []
    for i in range(len(matches)):
        sample = build_sample(matches,i,n)
        if sample is not None:
            TX , y = sample
            TX_list.append(TX)
            y_list.append(y)
    X = torch.stack(TX_list)
    Y = torch.stack(y_list)
    return X, Y