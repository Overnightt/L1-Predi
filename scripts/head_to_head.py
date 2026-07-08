#Compute a head-to-head (H2H) score between two teams for a given match date.
#Looks at all past confrontations between Hteam and Ateam (regardless of who was hosting), 
#and returns a score in [-1, 1]:
#    +1: Hteam has historically won against Ateam
#    -1: Ateam has historically won against Hteam
#     0: balanced record, or not enough history to tell
def h2h_score(Hteam : str ,Ateam: str, date: pd.Timestamp, matches: pd.DataFrame) :
    h2h_matches = matches[
        (((matches["HomeTeam"] == Hteam) & (matches["AwayTeam"] == Ateam)) | ((matches["HomeTeam"] == Ateam) & (matches["AwayTeam"] == Hteam))) & (matches["Date"] < date)
]

    W_Hteam = 0
    W_Ateam = 0
    for i, row in h2h_matches.iterrows():
        if row["HomeTeam"] == Hteam and row["FTR"] == "H":
            W_Hteam += 1
        elif row["HomeTeam"] == Ateam and row["FTR"] == "H":
            W_Ateam += 1
        elif row["AwayTeam"] == Hteam and row["FTR"] == "A":
            W_Hteam += 1
        elif row["AwayTeam"] == Ateam and row["FTR"] == "A":
            W_Ateam += 1

    total = len(h2h_matches) 
    if total == 0: 
        score = 0 # default to neutral (0) when the two teams have never played each other before
    else:
        score = (W_Hteam - W_Ateam) / total
    confidence = min(total/5, 1) # Shrink the score's weight when based on very few matches.
    score_h2h = score * confidence # Final H2H score
    return score_h2h
