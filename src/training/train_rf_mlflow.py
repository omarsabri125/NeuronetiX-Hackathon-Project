import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import mlflow
from models import train_random_forest
from data.preprocessing import preprocessing_pipline
from evaluation.metrics import evaluate_model
from utils.plotting import plot_confusion_matrix, plot_ROC_AUC
import argparse

FILE_PATH = "D:/Churn_Customer/NeuronetiX-Hackathon-Project/assets/Telecom_Customers _Churn_Dataset.csv"


def train_model(n_estimators, max_depth, use_smotten, plot_name):

    run_name = f"RandomForest_SMOTTEN_{use_smotten}"

    X_train, X_test, y_train, y_test = preprocessing_pipline(
        FILE_PATH, smotten=use_smotten)

    mlflow.set_experiment(f'churn-detection')
    with mlflow.start_run(run_name=run_name) as run:

        rf_model = train_random_forest(
            X_train, y_train, n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        y_pred = rf_model.predict(X_test)

        result = evaluate_model(y_test, y_pred)

        mlflow.log_params(
            {'n_estimators': n_estimators, 'max_depth': max_depth})
        mlflow.log_metrics(result)
        mlflow.sklearn.log_model(
            rf_model, artifact_path=f'{rf_model.__class__.__name__}_{plot_name}')

        plot_confusion_matrix(y_test, y_pred, plot_name)
        plot_ROC_AUC(y_test, y_pred, plot_name)


def main(n_estimators: int, max_depth: int):
    # 1. without considering the imabalancing data
    train_model(n_estimators=n_estimators, max_depth=max_depth,
                use_smotten=False, plot_name="without_smotten")

    # 2. with considering the imabalancing data using smotten
    train_model(n_estimators=n_estimators, max_depth=max_depth,
                use_smotten=True, plot_name="with_smotten")


if __name__ == '__main__':
    # Take input from user via CLI using argparser library
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', '-n', type=int, default=350)
    parser.add_argument('--max_depth', '-d', type=int, default=15)
    args = parser.parse_args()

    main(n_estimators=args.n_estimators, max_depth=args.max_depth)
