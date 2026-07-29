# [PathRIR: Physics-Guided Acoustic Path Selection and Late-Tail Compensation for Fast Room Impulse Response Simulation](https://arxiv.org/abs/2607.23293) (IWAENC 2026)

Official implementation and pretrained models for our IWAENC 2026 paper.

PathRIR accelerates high-order image source method (ISM) simulation by pruning acoustically unimportant reflection paths. A lightweight Compensation-MLP restores the missing late-reverberation energy.

---

## Paper

If you use this repository in your research, please cite:

```bibtex
@inproceedings{xu2026pathrir,
  author    = {Xu, Shaoheng and Sun, Chunyi and Zhang, Jihui and
               Bastine, Amy and Samarasinghe, Prasanga N. and
               Abhayapala, Thushara D.},
  title     = {{PathRIR}: Physics-Guided Acoustic Path Selection and Late-Tail
               Compensation for Fast Room Impulse Response Simulation},
  booktitle = {Proceedings of the 19th International Workshop on Acoustic
               Signal Enhancement (IWAENC)},
  year      = {2026}
}
```

* [arXiv paper](https://arxiv.org/abs/2607.23293)
* [Personal academic website](https://shaohenry.github.io/)

---

## Overview

* If you only want to try PathRIR to generate RIRs, install the package and use the example below.
* The remaining scripts are for rebuilding the datasets, training the models, and reproducing the evaluation.
* The bundled checkpoints use a maximum reflection order of 10, an 8 kHz sampling rate, and 0.5-second RIRs.
* Generated datasets are not included because the `.npz` files are large, but they can be recreated using the provided scripts.

---

## Repository contents

| Path                            | Description                                       |
| ------------------------------- | ------------------------------------------------- |
| `pathrir/`                      | Installable PathRIR package                       |
| `polygon_ism_engine.py`         | Incremental ISM engine for extruded polygon rooms |
| `build_ism_pruning_dataset.py`  | Dataset generation and pruning-label construction |
| `train_ism_pruning_mlp.py`      | Pruning-MLP training                              |
| `train_edc_compensation_mlp.py` | Compensation-MLP training                         |
| `evaluate_pathrir.py`           | Evaluation, timing, metrics, and RIR export       |
| `example_commands.txt`          | Data, training, and evaluation commands           |
| `checkpoints/`                  | Pretrained order-10 checkpoints                   |
| `data/`                         | Default dataset directory                         |

---

## Installation

PathRIR requires Python 3.9 or later.

```bash
git clone https://github.com/ShaoHenry/PathRIR.git
cd PathRIR
```

For pretrained inference:

```bash
python -m pip install .
```

For dataset generation, training, evaluation, and WAV export:

```bash
python -m pip install ".[full]"
```

The evaluation scripts were tested with `pyroomacoustics==0.7.7`. A GPU is optional. If you need a CUDA-enabled build of PyTorch, install it before installing PathRIR.

---

## Quick start

```python
from pathrir import PathRIR

simulator = PathRIR()

rir = simulator.simulate(
    corners=[
        [0.0, 0.0],
        [5.0, 0.0],
        [6.0, 3.0],
        [3.0, 5.0],
        [0.0, 4.0],
    ],
    height=3.0,
    absorption=0.3,
    source=[2.0, 2.0, 1.5],
    mics=[
        [3.0, 3.0, 1.2],
        [1.5, 3.2, 1.6],
    ],
    fs=8000,
    duration=0.5,
    max_order=10,
)

print(rir.shape)  # (2, 4000)
```

Geometry and positions are measured in metres. `corners` defines the 2-D floor plan and may be listed clockwise or counter-clockwise.

`absorption` accepts:

* one value for all surfaces;
* one value per wall, with the wall mean used for the floor and ceiling; or
* one value per wall followed by separate floor and ceiling values.

Set `compensate=False` to return the pruning-only RIR. Set `return_pruned=True` to return both `(compensated_rir, pruned_rir)`.

Custom checkpoints can be loaded with:

```python
simulator = PathRIR(
    pruning_ckpt="path/to/pruning_checkpoint.pt",
    compensation_ckpt="path/to/compensation_checkpoint.pt",
)
```

New checkpoints are recommended when the room distribution, sampling rate, RIR duration, or maximum reflection order differs from the bundled settings.

---

## Training and evaluation

See [example_commands.txt](example_commands.txt) for the complete command sequence. It covers:

1. dataset generation;
2. Pruning-MLP and Compensation-MLP training;
3. model evaluation; and
4. reflection-order evaluation from order 1 to 10.

High-order full-ISM simulation can require substantial memory and computation time. Test one room with one worker before starting a full run.

For timing comparisons, leave the machine otherwise idle, use at least three repeats, and add `--no-mem-profiling`.

Results are saved as per-room metrics, summary tables, and JSON files. Add `--save-rir-wavs` to export WAV files.

---

## License

PathRIR is released under the MIT License. See [LICENSE](LICENSE).
