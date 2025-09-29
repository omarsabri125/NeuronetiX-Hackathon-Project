import pandas as pd

def load_csv_file(file_path):

    df = pd.read_csv(file_path)
    return df

def load_excel_file(file_path):

    df = pd.read_excel(file_path)
    return df


