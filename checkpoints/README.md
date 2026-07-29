# Pretrained checkpoints

These are the order-10 models used for the paper experiments:

```text
checkpoints/
├── pruning_mlp/
│   └── best_by_safe_recall.pt
└── comp_mlp/
    └── best_by_edc_loss.pt
```

They were trained on 1,000 Monte Carlo-generated extruded polygon rooms with
an 8 kHz sampling rate and 0.5-second RIRs. The same files are bundled in
`pathrir/checkpoints/`, so `PathRIR()` can load them automatically after
installation.

The training scripts save new checkpoints in the directory given by
`--out-dir`. If you change the room distribution, sampling rate, RIR duration,
or maximum reflection order, train both models again and pass their paths to
`PathRIR`.
