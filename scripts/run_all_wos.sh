#!/usr/bin/env bash
set -euo pipefail

python src/data/preprocess_dummy.py --dataset wos

MODELS=(
  configs/model/papers/hcl.yaml
  configs/model/papers/hpt.yaml
  configs/model/papers/hitin.yaml
  configs/model/papers/hybrid_embed.yaml
  configs/model/papers/hb2m.yaml
  configs/model/lse_hf_lt.yaml
)

for m in "${MODELS[@]}"; do
  python src/run_train.py --config configs/train/default.yaml --data configs/data/wos.yaml --model "$m"
  python src/run_eval.py --config configs/train/default.yaml --data configs/data/wos.yaml --model "$m"
done

python scripts/collect_results.py
set -e
python src/data/preprocess_dummy.py --dataset wos
python src/run_train.py --config configs/train/default.yaml --data configs/data/wos.yaml --model configs/model/lse_hf_lt.yaml
python src/run_eval.py --config configs/train/default.yaml --data configs/data/wos.yaml --model configs/model/lse_hf_lt.yaml
