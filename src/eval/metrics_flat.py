import numpy as np
from sklearn.metrics import f1_score


def compute_micro_macro(y_true, y_pred):
    micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return {'micro_f1': float(micro), 'macro_f1': float(macro)}


def to_numpy_binary(pred_probs, threshold=0.5):
    return (pred_probs >= threshold).astype(np.int32)
