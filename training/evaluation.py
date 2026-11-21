"""Evaluation helpers for labelled sentence-pair experiments."""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
def evaluate(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r,
            "f1": f, "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()}
