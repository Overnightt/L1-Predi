import pandas as pd

df = pd.read_csv("..\data\season-2425.csv",delimiter=";")
df["Date"] = pd.to_datetime(df["Date"],dayfirst=True)
print(df.head())
print(df.dtypes)