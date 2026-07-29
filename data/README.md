# Generated data

This folder is reserved for locally generated training and test sets. The
datasets are not included in the repository because the compressed `.npz`
files can be large.

Use the commands below to recreate them.

```bash
# Training set: 1,000 rooms
python build_ism_pruning_dataset.py \
  --out-dir ./data/train_order10 \
  --num-configs 1000 \
  --max-order 10 \
  --fs 8000 \
  --rir-duration 0.5 \
  --num-mics 2 \
  --label-eps 1e-4 \
  --chain-blind-limit 4 \
  --dead-negatives-ratio 1.0 \
  --seed 0

# Test set: 20 rooms
python build_ism_pruning_dataset.py \
  --out-dir ./data/iwaenc_testset_order10 \
  --num-configs 20 \
  --max-order 10 \
  --fs 8000 \
  --rir-duration 0.5 \
  --num-mics 2 \
  --label-eps 1e-4 \
  --chain-blind-limit 4 \
  --dead-negatives-ratio 1.0 \
  --seed 6
```

Each room is stored in its own `.npz` file. The same output directory also
contains `manifest.jsonl`, `dataset_config.json`, and `summary.json`.

Room generation can run in parallel with `--num-workers N`. Start with one
worker, then increase the number only if the machine has enough free memory.
High-order ISM generation can be memory intensive.
