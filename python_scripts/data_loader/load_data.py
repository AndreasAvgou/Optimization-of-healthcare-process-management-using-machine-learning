import pandas as pd

def load_data(file_name, sheet_name):
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    if df.isnull().values.any():
        print("Missing values detected. Handling using mean imputation...")
        df.fillna(df.mean(), inplace=True)
    return df
