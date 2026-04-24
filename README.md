# Hierarchical Text Classification Thesis Project (RCV1 + WOS)

本项目用于本科毕业设计：**基于标签语义增强的层次文本分类设计与实现**。

## 功能
- 统一 RCV1/WOS 数据预处理接口
- 统一训练与评估流程（Micro/Macro F1 + Tail F1）
- 可扩展的模型注册机制（支持复现文献模型与自定义模型）
- 提供可视化系统骨架（FastAPI + Frontend 占位）

## 快速开始
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) 生成示例数据（占位）
python src/data/preprocess_dummy.py --dataset rcv1
python src/data/preprocess_dummy.py --dataset wos

# 2) 训练
python src/run_train.py --config configs/train/default.yaml --data configs/data/rcv1.yaml --model configs/model/lse_hf_lt.yaml

# 3) 评估
python src/run_eval.py --pred results/preds_rcv1.json --gold data/processed/rcv1/test.jsonl
```

## 说明
- 当前仓库提供可运行的**基线工程骨架**，便于你在此基础上接入 5 篇论文的具体实现。
- `src/models/registry.py` 预留了模型入口。

