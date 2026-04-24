import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class MultiLabelTextDataset(Dataset):
    def __init__(self, rows, label2id, model_name='distilbert-base-uncased', max_len=128):
        self.rows = rows
        self.label2id = label2id
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        tok = self.tokenizer(
            row['text'],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        y = torch.zeros(len(self.label2id), dtype=torch.float)
        for lb in row['labels']:
            if lb in self.label2id:
                y[self.label2id[lb]] = 1.0
        return {
            'input_ids': tok['input_ids'].squeeze(0),
            'attention_mask': tok['attention_mask'].squeeze(0),
            'labels': y,
        }
