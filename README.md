# DARF RF Fingerprinting

Device Analog/RF Fingerprint Classification.

## Goal

Build a robust neural network architecture that consistently classifies wireless transmitters using unique RF imperfections.

## Pipeline

Synthetic I/Q -> Dataset Loader -> CNN Classifier -> Metrics + Robustness Testing

## Installation

From the repository root, create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

If you are using PyCharm, select `.venv/bin/python` as the project interpreter after installing the dependencies.

## Week 1

Verify the toy signal generator:

```bash
python3 -m src.toy_generator
```

Verify the dataset loader:

```bash
python3 test_dataset.py
```

Train the baseline CNN and save metrics:

```bash
python3 -m src.training.train
```

The training command writes the Week 1 metrics file here:

```text
results/week1_toy_cnn/metrics.json
```

## Week 1

- Synthetic generator
- Dataset loader
- CNN baseline
- Evaluation pipeline
