import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from .loader import load_csv_file
from imblearn.combine import SMOTEENN
from pathlib import Path
import os
import pickle
import logging
import yaml
import json
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parents[2]

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
    scaler = RobustScaler()
    num_cols, _ = retrieve_num_cols(df_train)
    df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
    df_test[num_cols] = scaler.transform(df_test[num_cols])
    return df_train, df_test, scaler

def log_preprocessing_stats(df, X_train, X_test, y_train, y_test, output_file='data/processed/stats.json'):

    output_file = BASE_DIR / output_file
    os.makedirs(output_file.parent, exist_ok=True)

    stats = {
        'raw_data': {
            'total_samples': len(df),
            'total_features': len(df.columns),
            'missing_values': int(check_null(df))
        },
        'processed_data': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': len(X_train.columns),
            'train_churn_rate': float(y_train.mean()),
            'test_churn_rate': float(y_test.mean())
        },
        'feature_names': X_train.columns.tolist()
    }
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    return stats

def save_processed_data(X_train, X_test, y_train, y_test, scaler, encoders, output_dir='data/processed', smoteenn_applied=False):

    # Create subfolder based on whether SMOTEENN was applied
    if smoteenn_applied:
        output_dir = BASE_DIR / output_dir / 'with_smoteenn'
        models_dir = BASE_DIR / 'models' / 'with_smoteenn'
    else:
        output_dir = BASE_DIR / output_dir / 'without_smoteenn'
        models_dir = BASE_DIR / 'models' / 'without_smoteenn'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Save train/test data
    X_train.to_csv(output_dir / 'X_train.csv', index=False)
    X_test.to_csv(output_dir / 'X_test.csv', index=False)
    y_train.to_csv(output_dir / 'y_train.csv', index=False)
    y_test.to_csv(output_dir / 'y_test.csv', index=False)

    # Save scaler
    with open(models_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save label encoders
    with open(models_dir / 'label_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    logger.info(f"Saved processed data to {output_dir}")
    logger.info(f"SMOTEENN applied: {smoteenn_applied}")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

def load_processed_data(input_dir='data/processed', smoteenn_applied=False):

    # Load from appropriate subfolder
    if smoteenn_applied:
        input_dir = BASE_DIR / input_dir / 'with_smoteenn'
        models_dir = BASE_DIR / 'models' / 'with_smoteenn'
    else:
        input_dir = BASE_DIR / input_dir / 'without_smoteenn'
        models_dir = BASE_DIR / 'models' / 'without_smoteenn'
    
    X_train = pd.read_csv(input_dir / 'X_train.csv')
    X_test = pd.read_csv(input_dir / 'X_test.csv')
    y_train = pd.read_csv(input_dir / 'y_train.csv').values.ravel()
    y_test = pd.read_csv(input_dir / 'y_test.csv').values.ravel()
    
    with open(models_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(models_dir / 'label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    return X_train, X_test, y_train, y_test, scaler, encoders

def preprocessing_pipline(smotten=True):

    params_file = BASE_DIR / 'params.yaml'

    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    
    prepare_params = params.get('prepare', {})
    input_file = prepare_params.get('input_file')
    smotten = smotten if smotten is not None else prepare_params.get('smotten')
    test_size = prepare_params.get('test_size')
    random_state = prepare_params.get('random_state')
    drop_columns = prepare_params.get('drop_columns')
    target_column = prepare_params.get('target_column')

    df = load_csv_file(input_file)
    df = clean_and_convert(df)
    df_clean = handling_missing_vaules(df)
    df, encoders = label_encoding(df_clean)
    X = df.drop(columns=drop_columns + [target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = data_spliting(X, y, test_size, random_state, stratify=y)

    if smotten:
        X_train, y_train = data_balancing(X_train, y_train)
        logger.info("SMOTEENN applied for data balancing")

    X_train, X_test, scaler = scaling(X_train, X_test)
    
    # Save with appropriate folder name
    save_processed_data(X_train, X_test, y_train, y_test, scaler, encoders, smoteenn_applied=smotten)

    # Log stats with appropriate file name
    if smotten:
        stats_file = 'data/processed/with_smoteenn/stats.json'
    else:
        stats_file = 'data/processed/without_smoteenn/stats.json'
    
    log_preprocessing_stats(df, X_train, X_test, y_train, y_test, output_file=stats_file)
    
    logger.info(f"Preprocessing pipeline completed with SMOTEENN={'enabled' if smotten else 'disabled'}")


if __name__ == "__main__":

    logger.info("=" * 50)
    logger.info("Running preprocessing WITH SMOTEENN")
    logger.info("=" * 50)
    preprocessing_pipline(smotten=True)
    logger.info("=" * 50)
    logger.info("Running preprocessing WITHOUT SMOTEENN")
    logger.info("=" * 50)
    preprocessing_pipline(smotten=False)