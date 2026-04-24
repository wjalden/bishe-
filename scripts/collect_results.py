import glob
import json

rows = []
for p in sorted(glob.glob('results/*_test_metrics.json')):
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    rows.append((p, d.get('micro_f1'), d.get('macro_f1')))

print('file\tmicro\tmacro')
for r in rows:
    print(f'{r[0]}\t{r[1]:.4f}\t{r[2]:.4f}')
