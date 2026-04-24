import numpy as np


def hierarchical_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray):
    # Simplified hierarchical metric placeholder: identical to micro P/R/F1 for now.
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    p = float(tp / (tp + fp + 1e-12))
    r = float(tp / (tp + fn + 1e-12))
    f1 = float(2 * p * r / (p + r + 1e-12))
    return {'h_precision': p, 'h_recall': r, 'h_f1': f1}
