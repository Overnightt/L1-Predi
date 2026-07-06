def h2h_score(Hteam : str ,Ateam: str, date: pd.Timestamp, matches: pd.DataFrame) :
    h2h_matches = matches[
        (((matches["HomeTeam"] == Hteam) & (matches["AwayTeam"] == Ateam)) | ((matches["HomeTeam"] == Ateam) & (matches["AwayTeam"] == Hteam))) & (matches["Date"] < date)
]