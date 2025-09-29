import pandas as pd

def load_csv_file(file_path):

    df = pd.read_csv(file_path)
    return df

def load_excel_file(file_path):

    df = pd.read_excel(file_path)
    return df

if __name__ == "__main__":
    df = load_csv_file("D:/Churn_Customer/NeuronetiX-Hackathon-Project/assets/Telecom_Customers _Churn_Dataset.csv")
    print(df.select_dtypes(include='number').columns)
    mask_spaces = df['TotalCharges'].str.strip() == ""
    print(df[mask_spaces])
