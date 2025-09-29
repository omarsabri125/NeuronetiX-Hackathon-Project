from sklearn.ensemble import RandomForestClassifier
import pickle

def train_random_forest(X_train, y_train, **kwargs):
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model

def save_model(model, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

def load_model(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
