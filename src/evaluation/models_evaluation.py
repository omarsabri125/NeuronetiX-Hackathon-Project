from pathlib import Path
import pandas as pd
import pickle
from .metrics import evaluate_model
from ..utils.log_metrics import log_metrics
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parents[2]

def load_test_set(input_dir='data/processed', smoteenn_applied=False):

    if smoteenn_applied:
        input_dir = BASE_DIR / input_dir / 'with_smoteenn'
    else:
        input_dir = BASE_DIR / input_dir / 'without_smoteenn'
    
    X_test = pd.read_csv(input_dir / 'X_test.csv')
    y_test = pd.read_csv(input_dir / 'y_test.csv').values.ravel()

    return X_test, y_test

def load_specific_model(model_path):

    with open (model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model 

def predictions_pipeline(model_path, smoteenn_applied):
    
    X_test, y_test = load_test_set(smoteenn_applied=smoteenn_applied)

    model = load_specific_model(model_path)

    y_pred = model.predict(X_test)

    results = evaluate_model(y_test, y_pred)

    log_metrics(model_path.name, smoteenn_applied, results)

def run_all_models(models_dir='models'):
    for smoteenn in [True, False]:
        if smoteenn:
            dir_path = BASE_DIR / models_dir / 'with_smoteenn'
        else:
            dir_path = BASE_DIR / models_dir / 'without_smoteenn'

        model_files = [
            f for f in dir_path.glob("*.pkl")
            if f.name not in ["scaler.pkl", "label_encoders.pkl"]
        ]
        logger.info(f"\nRunning models in {dir_path} (smoteenn={smoteenn})")

        for model_file in model_files:
            logger.info(f" -> Running {model_file.name} ...")
            results = predictions_pipeline(model_file, smoteenn)
            logger.info(f"    Results: {results}")


if __name__ == "__main__":
    run_all_models()







