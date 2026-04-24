import argparse
import json
from pathlib import Path
import random


def build_dummy(dataset: str):
    base = Path(f"data/processed/{dataset}")
    base.mkdir(parents=True, exist_ok=True)
    labels = [f"L{i}" for i in range(8)]
    splits = [("train", 120), ("val", 20), ("test", 30)]

    for split, n in splits:
        with open(base / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for i in range(n):
                k = 1 if random.random() < 0.7 else 2
                y = random.sample(labels, k)
                row = {"id": f"{split}_{i}", "text": f"sample text {i} for {dataset}", "labels": y}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    hierarchy = {
        "root": "ROOT",
        "parent2children": {"ROOT": ["L0", "L1", "L2", "L3"], "L0": ["L4", "L5"], "L1": ["L6", "L7"]}
    }
    with open(base / "hierarchy.json", "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rcv1", "wos"], required=True)
    args = parser.parse_args()
    build_dummy(args.dataset)
