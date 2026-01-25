import pandas as pd

# Returns the last `n` matches played by `team` strictly before `date`, ordered chronologically 
def build_sequence(team: str,date: pd.Timestamp,matches: pd.DataFrame,n: int):
    team_matches = matches [
        ((matches["HomeTeam"]==team )| (matches["AwayTeam"]==team)) & (matches["Date"]<date)
    ].sort_values("Date").tail(n)
    return team_matches
