# PROBE: Perturbation-Aware Structural Probing with Causal Subspace Isolation for Semi-Supervised Graph Anomaly Detection

This repository contains the implementation of **PROBE**, a semi-supervised graph anomaly detection framework that combines perturbation-aware structural probing, uncertainty-driven soft pseudo-labeling, and causal subspace isolation.

---

## Getting Started

### Setup Environment

The code was tested in the following environment:

| Component | Version |
|---|---|
| NVIDIA Driver | 560.35.03 |
| CUDA | 12.6 |
| Python | 3.9.25 |
| PyTorch | 2.4.0+cu124 |
| DGL | 2.4.0+cu124 |
| NumPy | 2.0.2 |
| scikit-learn | 1.6.1 |

A Conda environment can be created as follows:

```bash
conda create -n probe python=3.9.25 -y
conda activate probe
```

Install PyTorch and DGL builds compatible with CUDA 12.4, followed by the remaining dependencies required by the code.

---

### Preparing Datasets

The current implementation supports the following ten graph anomaly detection datasets:

`Reddit`, `Weibo`, `Amazon`, `YelpChi`, `T-Finance`, `Elliptic`, `Tolokers`, `Questions`, `DGraph-Fin`, and `T-Social`.

The processed datasets follow the [GADBench](https://github.com/squareRoot3/GADBench) data format. After downloading and preprocessing the datasets, place the graph files in the `datasets` directory:

```text
PROBE/
├── datasets/
│   ├── reddit
│   ├── weibo
│   ├── amazon
│   ├── yelp
│   ├── tfinance
│   ├── elliptic
│   ├── tolokers
│   ├── questions
│   ├── dgraphfin
│   └── tsocial
├── main.py
└── ...
```

Each processed DGL graph is expected to contain the following node data fields:

- `feature`: node attribute matrix;
- `label`: binary anomaly labels;
- `train_masks`: training split masks;
- `val_masks`: validation split masks;
- `test_masks`: test split masks.

DGraph-Fin and Elliptic may need to be downloaded and preprocessed separately according to their original release requirements. Their processed filenames must match the dataset names used by `main.py`.

---

### Model Configuration

The default hyperparameters for semi-supervised experiments are provided in:

```text
semi_train.conf.yaml
```

---

### Training and Evaluation

Run a semi-supervised experiment on Reddit with the default configuration:

```bash
python main.py --dataset reddit
```

Run the experiment on a specified GPU and configuration file:

```bash
python main.py \
  --dataset reddit \
  --device 0 \
  --config_path semi_train.conf.yaml
```

The supported dataset arguments are:

```text
reddit, weibo, amazon, yelp, tfinance,
elliptic, tolokers, questions, dgraphfin, tsocial
```

The training script reports the following evaluation metrics:

- **AUROC**;
- **AUPRC**;
- **Recall@K**, where `K` is the number of anomalous nodes in the evaluated split.

---

## Acknowledgements

The dataset organization and evaluation protocol are compatible with [GADBench](https://github.com/squareRoot3/GADBench). We thank the authors of GADBench and the open-source graph anomaly detection community for making their datasets and implementations publicly available.
