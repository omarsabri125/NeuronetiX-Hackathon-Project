import mlflow
from models import train_xgboost
from data.preprocessing import preprocessing_pipline
from evaluation.metrics import evaluate_model
from utils.plotting import plot_confusion_matrix, plot_ROC_AUC
import argparse

FILE_PATH = "D:/Churn_Customer/NeuronetiX-Hackathon-Project/assets/Telecom_Customers _Churn_Dataset.csv"


def train_model(n_estimators, max_depth, learning_rate, use_smotten, plot_name):

    run_name = f"XGBOOST_SMOTTEN_{use_smotten}"

    X_train, X_test, y_train, y_test = preprocessing_pipline(
        FILE_PATH, smotten=use_smotten)

    mlflow.set_experiment(f'churn-detection')
    with mlflow.start_run(run_name=run_name) as run:

        xgboost_model = train_xgboost(X_train, y_train, objective='binary:logistic', n_estimators=n_estimators,
                                      learning_rate=learning_rate,
                                      max_depth=max_depth)

        y_pred = xgboost_model.predict(X_test)

        result = evaluate_model(y_test, y_pred)

        mlflow.log_params({'n_estimators': n_estimators,
                          'learning_rate': learning_rate, 'max_depth': max_depth})
        mlflow.log_metrics(result)
        mlflow.xgboost.log_model(
            xgboost_model, artifact_path=f'{xgboost_model.__class__.__name__}_{plot_name}')

        plot_confusion_matrix(y_test, y_pred, plot_name)
        plot_ROC_AUC(y_test, y_pred, plot_name)


def main(n_estimators: int, max_depth: int, learning_rate: float):
    # 1. without considering the imabalancing data
    train_model(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                use_smotten=False, plot_name="without_smotten")

    # 2. with considering the imabalancing data using smotten
    train_model(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                use_smotten=True, plot_name="with_smotten")


if __name__ == '__main__':
    # Take input from user via CLI using argparser library
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', '-n', type=int, default=350)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1)
    parser.add_argument('--max_depth', '-d', type=int, default=15)
    args = parser.parse_args()

    main(n_estimators=args.n_estimators, max_depth=args.max_depth,
         learning_rate=args.learning_rate)
    
