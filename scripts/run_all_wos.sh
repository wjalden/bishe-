#!/usr/bin/env bash
set -e
python src/data/preprocess_dummy.py --dataset wos
python src/run_train.py --config configs/train/default.yaml --data configs/data/wos.yaml --model configs/model/lse_hf_lt.yaml
python src/run_eval.py --config configs/train/default.yaml --data configs/data/wos.yaml --model configs/model/lse_hf_lt.yaml
