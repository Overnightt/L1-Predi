import pandas as pd

# Computes normalization statistics (mean and standard deviation) for AvgAge and SquadValue
# Returns a dict: {"age_mean", "age_std", "value_mean", "value_std"}

def norm_stats(train_matches: pd.DataFrame, squad_values: pd.DataFrame):
    max_date = train_matches["Date"].max()
    filtered = squad_values[squad_values["SeasonStart"] <= max_date]
    age_mean = filtered["AvgAge"].mean()        # computes mean
    age_std = filtered["AvgAge"].std()          # computes standard deviation
    value_mean = filtered["SquadValue"].mean()  # computes mean
    value_std = filtered["SquadValue"].std()    # computes standard deviation
    return {
    "age_mean": age_mean,
    "age_std": age_std,
    "value_mean": value_mean,
    "value_std": value_std
}