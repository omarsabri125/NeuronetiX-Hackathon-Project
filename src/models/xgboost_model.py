import xgboost as xgb
import pickle

def train_xgboost(X_train, y_train, **kwargs):
    model = xgb.XGBClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model

def save_model(model, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

def load_model(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
