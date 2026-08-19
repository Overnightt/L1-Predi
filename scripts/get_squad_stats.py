import pandas as pd

# Extract the average age and the squad value from a specific squad, on a specific season 
def get_squad_stats(team: str, date: pd.Timestamp, teams_table: pd.DataFrame):
    squad_stats = teams_table[((teams_table["Team"] == team) & (teams_table["SeasonStart"] <= date))]
    squad_stats = squad_stats.sort_values("SeasonStart",ascending = False)
    if squad_stats.empty:
        return None
    latest_stats = squad_stats.iloc[0]
    return latest_stats["AvgAge"], latest_stats["SquadValue"]