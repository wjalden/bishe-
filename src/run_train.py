import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.io import load_yaml, read_jsonl, write_json
from src.utils.seed import set_seed
from src.data.dataset import MultiLabelTextDataset
from src.models.registry import build_model
from src.losses.focal import focal_bce_with_logits
from src.losses.hierarchy_consistency import hierarchy_consistency_loss
from src.eval.metrics_flat import compute_micro_macro, to_numpy_binary


def build_label_map(rows):
    labels = sorted({lb for r in rows for lb in r['labels']})
    return {lb: i for i, lb in enumerate(labels)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()

    cfg_t = load_yaml(args.config)
    cfg_d = load_yaml(args.data)
    cfg_m = load_yaml(args.model)

    set_seed(cfg_t['seed'])
    device = torch.device(cfg_t['device'])

    train_rows = read_jsonl(cfg_d['train_path'])
    val_rows = read_jsonl(cfg_d['val_path'])
    label2id = build_label_map(train_rows + val_rows)
    cfg_m['num_labels'] = len(label2id)

    train_ds = MultiLabelTextDataset(train_rows, label2id, cfg_m['text_encoder'], cfg_t['max_len'])
    val_ds = MultiLabelTextDataset(val_rows, label2id, cfg_m['text_encoder'], cfg_t['max_len'])
    train_loader = DataLoader(train_ds, batch_size=cfg_t['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg_t['batch_size'])

    model = build_model(cfg_m).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg_t['lr'], weight_decay=cfg_t['weight_decay'])

    parent_child_pairs = [(0, 4), (0, 5), (1, 6), (1, 7)] if len(label2id) >= 8 else []

    best = -1
    best_path = f"{cfg_t['save_dir']}/{cfg_d['name']}_{cfg_m['name']}.pt"

    for epoch in range(cfg_t['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}")
        for batch in pbar:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            y = batch['labels'].to(device)

            logits = model(ids, mask)
            cls_loss = focal_bce_with_logits(logits, y)
            hier_loss = hierarchy_consistency_loss(logits, parent_child_pairs)
            loss = cls_loss + cfg_t['lambda_hier'] * hier_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                y = batch['labels'].cpu().numpy()
                prob = torch.sigmoid(model(ids, mask)).cpu().numpy()
                ys.append(y)
                ps.append(prob)

        import numpy as np
        y_true = np.concatenate(ys, axis=0)
        y_pred = to_numpy_binary(np.concatenate(ps, axis=0), 0.5)
        m = compute_micro_macro(y_true, y_pred)
        if m['macro_f1'] > best:
            best = m['macro_f1']
            torch.save({'state_dict': model.state_dict(), 'label2id': label2id, 'model_cfg': cfg_m}, best_path)

    write_json(f"results/{cfg_d['name']}_{cfg_m['name']}_train_metrics.json", {'best_macro_f1': best})
    print(f"saved: {best_path}")


if __name__ == '__main__':
    main()
