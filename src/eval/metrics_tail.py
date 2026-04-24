import numpy as np
from sklearn.metrics import f1_score


def split_head_mid_tail(label_freq: list[int], head_ratio=0.2, mid_ratio=0.3):
    idx = np.argsort(np.array(label_freq))[::-1]
    n = len(idx)
    h = int(n * head_ratio)
    m = int(n * (head_ratio + mid_ratio))
    return {
        'head': idx[:h].tolist(),
        'mid': idx[h:m].tolist(),
        'tail': idx[m:].tolist(),
    }


def grouped_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, groups: dict[str, list[int]]):
    out = {}
    for name, cols in groups.items():
        if not cols:
            out[f'{name}_macro_f1'] = 0.0
            continue
        out[f'{name}_macro_f1'] = float(
            f1_score(y_true[:, cols], y_pred[:, cols], average='macro', zero_division=0)
        )
    return out
