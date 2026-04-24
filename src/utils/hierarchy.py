import json
from pathlib import Path


def load_hierarchy(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_parent_child_pairs(hierarchy: dict, label2id: dict[str, int]) -> list[tuple[int, int]]:
    parent2children = hierarchy.get('parent2children', {})
    pairs = []
    for parent, children in parent2children.items():
        if parent not in label2id:
            continue
        p = label2id[parent]
        for child in children:
            if child in label2id:
                pairs.append((p, label2id[child]))
    return pairs


def infer_label_freq(rows: list[dict], label2id: dict[str, int]) -> list[int]:
    counts = [0] * len(label2id)
    for row in rows:
        for lb in row.get('labels', []):
            if lb in label2id:
                counts[label2id[lb]] += 1
    return counts


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
