from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except:
        roc_auc = None
    
    return {
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    }
