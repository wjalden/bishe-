import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.io import load_yaml, read_jsonl, write_json
from src.data.dataset import MultiLabelTextDataset
from src.models.registry import build_model
from src.eval.metrics_flat import compute_micro_macro, to_numpy_binary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()

    cfg_t = load_yaml(args.config)
    cfg_d = load_yaml(args.data)
    cfg_m = load_yaml(args.model)

    ckpt_path = f"{cfg_t['save_dir']}/{cfg_d['name']}_{cfg_m['name']}.pt"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    label2id = ckpt['label2id']
    cfg_m = ckpt['model_cfg']

    test_rows = read_jsonl(cfg_d['test_path'])
    test_ds = MultiLabelTextDataset(test_rows, label2id, cfg_m['text_encoder'], cfg_t['max_len'])
    test_loader = DataLoader(test_ds, batch_size=cfg_t['batch_size'])

    model = build_model(cfg_m)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for batch in test_loader:
            prob = torch.sigmoid(model(batch['input_ids'], batch['attention_mask'])).cpu().numpy()
            ys.append(batch['labels'].cpu().numpy())
            ps.append(prob)

    y_true = np.concatenate(ys, axis=0)
    y_pred = to_numpy_binary(np.concatenate(ps, axis=0), 0.5)
    metrics = compute_micro_macro(y_true, y_pred)
    write_json(f"results/{cfg_d['name']}_{cfg_m['name']}_test_metrics.json", metrics)
    print(metrics)


if __name__ == '__main__':
    main()
