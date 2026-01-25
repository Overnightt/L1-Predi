import pandas as pd

# Loads a CSV file and ensures proper formatting: 
#    - Converts 'Date' column to datetime 
#    - Uses ';' as delimiter
def check_data(filepath):
    df = pd.read_csv(filepath,delimiter=";")
    df["Date"] = pd.to_datetime(df["Date"],dayfirst=True)
    return df