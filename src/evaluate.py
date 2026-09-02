"""Shared evaluation for all models"""

from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

def evaluate(y_true, y_pred) -> dict:
    """Evaluate macro f1, accuracy and weighted f1 based on the predicted and true labels.

    Args:
        y_true: an array of true labels of instances
        y_pred: an array of predicted labels of instances

    Returns:
        Dictionary with keys as macro f1, accuracy and weighted f1 and their values
    """
    result_dict = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    return result_dict

def full_report(y_true, y_pred, labels=None, target_names=None):
    """Print per-class classification report and return confusion matrix.

    Args:
        y_true: an array of true labels of instances
        y_pred: an array of predicted labels of instances

    Returns:
        Confusion matrix of size NUM_CLASSES x NUM_CLASSES 
    """
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=target_names, zero_division=0))
    return confusion_matrix(y_true, y_pred, labels=labels)