import pandas as pd

# Loads a CSV file and ensures proper formatting: 
#    - Converts 'Date' column to datetime 
#    - Uses ';' as delimiter
def check_data(filepath,date_column="Date"):
    df = pd.read_csv(filepath,delimiter=";")
    df[date_column] = pd.to_datetime(df[date_column],dayfirst=True)  
    return df