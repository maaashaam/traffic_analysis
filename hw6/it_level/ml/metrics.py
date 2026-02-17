from __future__ import annotations

from sklearn.metrics import classification_report, confusion_matrix


def print_report(y_true, y_pred) -> None:
    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred, digits=3))

    print("=== Confusion matrix (rows=true, cols=pred) ===")
    labels = ["junior", "middle", "senior"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(labels)
    print(cm)
