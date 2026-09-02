"""Shared evaluation for all models"""

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

def evaluate(y_true, y_pred) -> dict:
    """Headline metrics"""
    result_dict = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    return result_dict

def full_report(y_true, y_pred, labels=None, target_names=None):
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=target_names, zero_division=0))
    return confusion_matrix(y_true, y_pred, labels=labels)