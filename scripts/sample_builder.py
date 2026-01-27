import torch
import pandas as pd
from build_sequences import build_sequence
from sequence_to_tensor import sequence_to_tensor

#Builds a training sample for a single match. Skips matches that don't have enough prior history.
def build_sample(matches: pd.DataFrame,row : int,n: int):
    match= matches.iloc[row]
    date= match["Date"]
    HT= match["HomeTeam"]
    AT= match["AwayTeam"]
    seqH = build_sequence(HT,date,matches,n)
    seqA = build_sequence(AT,date,matches,n)
    if len(seqH) <n or len(seqA)<n:
        print("the match is to early to have a meaningfull sample")
        return None
    TH = sequence_to_tensor(seqH)
    TA = sequence_to_tensor(seqA)
    TX = torch.cat((TH,TA),dim=1)
    outcome_map = {"H":0,"D":1,"A":2}
    y= torch.tensor([outcome_map[match["FTR"]]],dtype=torch.long)
    return TX,y