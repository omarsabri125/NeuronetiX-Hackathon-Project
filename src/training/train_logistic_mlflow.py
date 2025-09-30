import mlflow
from models import train_logistic_regression
from data.preprocessing import preprocessing_pipline
from evaluation.metrics import evaluate_model
from utils.plotting import plot_confusion_matrix, plot_ROC_AUC
import argparse

FILE_PATH = "D:/Churn_Customer/NeuronetiX-Hackathon-Project/assets/Telecom_Customers _Churn_Dataset.csv"


def train_model(C: float, penalty: str, use_smotten, plot_name):

    run_name = f"LogisticRegression_SMOTTEN_{use_smotten}"

    X_train, X_test, y_train, y_test = preprocessing_pipline(
        FILE_PATH, smotten=use_smotten)

    mlflow.set_experiment(f'churn-detection')
    with mlflow.start_run(run_name=run_name) as run:

        lr_model = train_logistic_regression(
            X_train, y_train, C=C, penalty=penalty, random_state=42)

        y_pred = lr_model.predict(X_test)

        result = evaluate_model(y_test, y_pred)

        mlflow.log_params(
            {'C': C, 'penalty': penalty})
        mlflow.log_metrics(result)
        mlflow.sklearn.log_model(
            lr_model, artifact_path=f'{lr_model.__class__.__name__}_{plot_name}')

        plot_confusion_matrix(y_test, y_pred, plot_name)
        plot_ROC_AUC(y_test, y_pred, plot_name)


def main(C: float, penalty: str):
    # 1. without considering the imabalancing data
    train_model(C=C, penalty=penalty,
                use_smotten=False, plot_name="without_smotten")

    # 2. with considering the imabalancing data using smotten
    train_model(C=C, penalty=penalty,
                use_smotten=True, plot_name="with_smotten")


if __name__ == '__main__':
    # Take input from user via CLI using argparser library
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', '-c', type=float, default=2.5)
    parser.add_argument('--penalty', '-p', type=str, default=None)
    args = parser.parse_args()

    # Call the main function
    main(C=args.C, penalty=args.penalty)

# python -m training.train_logistic_mlflow
