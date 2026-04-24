# Hierarchical Text Classification Thesis Project (RCV1 + WOS)

本项目用于本科毕业设计：**基于标签语义增强的层次文本分类设计与实现**。

## 当前实现范围
- 统一数据接口（RCV1/WOS 配置）
- 统一训练/评估入口
- 统一指标：Micro-F1 / Macro-F1 / h-F1 / Head-Mid-Tail Macro-F1
- 5 篇论文模型入口（HPT / HiTIN / HCL / Hybrid-Embed / HB2M）
- 你的模型入口（LSE-HF-LT）
- FastAPI 可视化后端占位接口

> 注意：论文模型当前为**统一框架复现实验脚手架**，便于快速跑通和横向比较；你可以在 `src/models/papers/` 中逐个替换成严格论文实现。

## 快速开始
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 生成演示数据（可替换为真实 RCV1/WOS 预处理）
python src/data/preprocess_dummy.py --dataset rcv1
python src/data/preprocess_dummy.py --dataset wos

# 跑通全部模型（rcv1）
bash scripts/run_all_rcv1.sh

# 跑通全部模型（wos）
bash scripts/run_all_wos.sh
```

## 目录说明
- `configs/model/papers/*.yaml`：5 篇论文入口配置
- `src/run_train.py`：统一训练主程序
- `src/run_eval.py`：统一评估主程序
- `src/eval/metrics_tail.py`：长尾分组评估
- `scripts/collect_results.py`：统一结果汇总

## 上传到你的 GitHub
```bash
git remote add origin https://github.com/wjalden/bishe-.git
git push -u origin HEAD
```

