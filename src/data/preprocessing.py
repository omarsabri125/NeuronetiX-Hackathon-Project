import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from .loader import load_csv_file
from imblearn.combine import SMOTEENN


def check_null(df: pd.DataFrame):
    return df.isnull().sum().sum()


def check_columns_type(df: pd.DataFrame):
    return df.dtypes


def clean_and_convert(df: pd.DataFrame):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)

        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
            if df[col].dropna().apply(float.is_integer).all():
                df[col] = df[col].astype('Int64')
            else:
                df[col] = df[col].astype(float)
        except Exception:
            df[col] = df[col].astype(object)

    return df


def retrieve_categ_cols(df: pd.DataFrame):
    categ_cols = df.select_dtypes(include='object').columns
    categ_cols = categ_cols.to_list()
    return categ_cols


def retrieve_num_cols(df: pd.DataFrame, threshold=6):
    num_cols = []
    num_categ_cols = []
    for col in df.select_dtypes(include='number').columns:
        if df[col].nunique() > threshold:
            num_cols.append(col)
        else:
            num_categ_cols.append(col)

    return num_cols, num_categ_cols


def handling_missing_vaules(df: pd.DataFrame):
    if check_null(df) == 0:
        return df
    categ_cols = retrieve_categ_cols(df)
    num_cols, num_categ_cols = retrieve_num_cols(df)
    # categorical + numeric-categorical
    for col in categ_cols + num_categ_cols:
        if df[col].isnull().sum() > 0:
            most_freq = df[col].mode()[0]
            df[col] = df[col].fillna(most_freq)

    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median)

    return df


def data_spliting(X, y, test_size=0.2, random_state=42, stratify=None):

    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state,
                            stratify=stratify)


def data_balancing(X_train, y_train):
    sm = SMOTEENN()
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    return X_train_res, y_train_res


def label_encoding(df: pd.DataFrame):
    encoders = {}
    categ_cols = retrieve_categ_cols(df)
    for col in categ_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform((df[col].astype(str)))
        encoders[col] = le

    return df, encoders


def scaling(df_train: pd.DataFrame, df_test: pd.DataFrame):
    scaler = StandardScaler()
    num_cols, _ = retrieve_num_cols(df_train)
    df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
    df_test[num_cols] = scaler.transform(df_test[num_cols])
    return df_train, df_test


def preprocessing_pipline(file_path, smotten=False):
    df = load_csv_file(file_path)
    df = clean_and_convert(df)
    df = handling_missing_vaules(df)
    df, _ = label_encoding(df)
    X = df.drop(columns=['customerID', 'PhoneService', 'gender',
                'StreamingTV', 'StreamingMovies', 'MultipleLines', 'Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = data_spliting(X, y, stratify=y)

    if smotten:
        X_train, y_train = data_balancing(X_train, y_train)

    X_train, X_test = scaling(X_train, X_test)
    return X_train, X_test, y_train, y_test


# X_train, X_test, y_train, y_test = preprocessing_pipline(
#     "D:/Churn_Customer/NeuronetiX-Hackathon-Project/assets/Telecom_Customers _Churn_Dataset.csv",smotten=True)

# print(X_train.head())
# print(X_test.head())
# print(y_train.head())
# print(y_test.head())
