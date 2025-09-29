import mlflow
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix(y_test, y_pred_test, plot_name):
    # Plot the confusion matrix and save it to mlflow
    plt.figure(figsize=(10, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_test),
                annot=True, cbar=False, fmt='.2f', cmap='Blues')
    plt.title(f'{plot_name}')
    plt.xticks(ticks=np.arange(2) + 0.5, labels=[False, True])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=[False, True])

    # Save the plot to MLflow
    conf_matrix_fig = plt.gcf()
    mlflow.log_figure(figure=conf_matrix_fig,
                      artifact_file=f"plots/{plot_name}_conf_matrix.png")
    plt.close()

def plot_ROC_AUC(y_test, y_pred_test, plot_name):
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_test)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve and save it to mlflow
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # Save the plot to MLflow
    roc_fig = plt.gcf()
    mlflow.log_figure(figure=roc_fig, artifact_file=f'plots/{plot_name}_roc_curve.png')
    plt.close()

