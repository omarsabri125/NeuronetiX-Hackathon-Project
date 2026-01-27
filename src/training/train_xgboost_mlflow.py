import mlflow
from ..models import train_xgboost, save_model
from ..data.preprocessing import load_processed_data
from ..evaluation.metrics import evaluate_model
from ..utils.plotting import plot_confusion_matrix, plot_ROC_AUC
import argparse

def train_model(n_estimators, max_depth, learning_rate, use_smotten, plot_name):

    run_name = f"XGBOOST_SMOTTEN_{use_smotten}"

    if use_smotten:
        X_train, X_test, y_train, y_test, _, _ = load_processed_data(smoteenn_applied=True)
    else:
        X_train, X_test, y_train, y_test, _, _ = load_processed_data(smoteenn_applied=False)

    mlflow.set_experiment(f'churn-detection')
    with mlflow.start_run(run_name=run_name) as run:

        xgboost_model = train_xgboost(X_train, y_train, objective='binary:logistic', n_estimators=n_estimators,
                                      learning_rate=learning_rate,
                                      max_depth=max_depth)
        
        save_model(xgboost_model, f'models/{plot_name}/xgboost_model.pkl')

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
                use_smotten=False, plot_name="without_smoteenn")

    # 2. with considering the imabalancing data using smotten
    train_model(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                use_smotten=True, plot_name="with_smoteenn")


if __name__ == '__main__':
    # Take input from user via CLI using argparser library
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', '-n', type=int, default=350)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1)
    parser.add_argument('--max_depth', '-d', type=int, default=15)
    args = parser.parse_args()

    main(n_estimators=args.n_estimators, max_depth=args.max_depth,
         learning_rate=args.learning_rate)
    
