import mlflow
from ..models import train_random_forest, save_model
from ..data.preprocessing import load_processed_data
from ..evaluation.metrics import evaluate_model
from ..utils.plotting import plot_confusion_matrix, plot_ROC_AUC
import argparse

def train_model(n_estimators, max_depth, use_smotten, plot_name):

    run_name = f"RandomForest_SMOTTEN_{use_smotten}"

    if use_smotten:
        X_train, X_test, y_train, y_test, _, _ = load_processed_data(smoteenn_applied=True)
    else:
        X_train, X_test, y_train, y_test, _, _ = load_processed_data(smoteenn_applied=False)

    mlflow.set_experiment(f'churn-detection')
    with mlflow.start_run(run_name=run_name) as run:

        rf_model = train_random_forest(
            X_train, y_train, n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        save_model(rf_model, f'models/{plot_name}/random_forest.pkl')

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
                use_smotten=False, plot_name="without_smoteenn")

    # 2. with considering the imabalancing data using smotten
    train_model(n_estimators=n_estimators, max_depth=max_depth,
                use_smotten=True, plot_name="with_smoteenn")


if __name__ == '__main__':
    # Take input from user via CLI using argparser library
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', '-n', type=int, default=350)
    parser.add_argument('--max_depth', '-d', type=int, default=15)
    args = parser.parse_args()

    main(n_estimators=args.n_estimators, max_depth=args.max_depth)
