# Network Intrusion Detection System (IDS) — Random Forest

A passive, machine-learning-based Intrusion Detection System that classifies network traffic as **benign or malicious**, built as a final-year university project. The system uses the **NSL-KDD** dataset to train a supervised Random Forest classifier, achieving **86.95% accuracy**.

> **Platform note:** The model training and data pipeline (`scripts/`) run on any platform with Python installed. **Live packet capture mode** specifically requires Linux (developed and tested on Ubuntu 24.04), since it depends on raw socket access via Scapy, typically alongside `sudo`. **File mode**, which analyses a `.pcap` file you already have, works cross-platform.

## Overview

This project develops a passive IDS designed to classify network traffic as benign or malicious. It uses labelled datasets such as NSL-KDD to train a supervised learning model capable of detecting abnormal activity based on statistical features, without actively interfering with the traffic it inspects.

Core functionality:
- Data preprocessing and transformation
- Feature selection (constant-feature removal, correlation analysis)
- Random Forest classification with class imbalance handling (SMOTE)
- Two detection modes: live capture and offline `.pcap` file analysis
- Logging of suspicious or malicious predictions

## Model Performance

- **Algorithm:** Random Forest (100 estimators)
- **Accuracy:** 86.95%
- **Dataset:** NSL-KDD (`KDDTrain+` / `KDDTest+`)
- **Classification threshold:** Tuned to 0.1 (lower than the default 0.5) to favour recall — catching more true attacks at the cost of some extra false positives, a reasonable trade-off for an IDS where missed attacks are more costly than false alarms.
- Models compared during development: Random Forest, Decision Tree, Logistic Regression (Random Forest performed best)

## How It Works

1. **Data preprocessing** (`preprocessing.py`) — removes constant/near-constant features (e.g. columns that are >99% zeros) and highly correlated duplicate features identified during EDA, then encodes and scales the remaining features.
2. **Model training** (`model_training.py`) — trains and cross-validates multiple classifiers with SMOTE oversampling to address class imbalance, selecting the best-performing model.
3. **Detection** (`detector.py`) — loads the trained model and either:
   - **Live mode:** sniffs real network traffic on a chosen interface using Scapy, extracting per-connection statistical features in real time
   - **File mode:** reads a `.pcap` file captured in Wireshark and replays the same feature extraction and classification pipeline offline
4. **Logging** — any connection classified as malicious above the threshold is written to `logs/detections.log` with its predicted probability.

## Requirements

- Python 3.12
- See [`requirements.txt`](requirements.txt) for exact package versions
- Live capture mode additionally requires Linux and typically root privileges (for raw socket access)

```bash
pip install -r requirements.txt
```

## Usage

Run the detector and follow the interactive prompts:

```bash
python scripts/detector.py
```

You'll be asked to choose:
- **1 — Live capture**: select a network interface and watch real-time classification (Linux, typically needs `sudo python scripts/detector.py`)
- **2 — File mode**: provide a path to a `.pcap` file for offline analysis

To retrain the model from scratch:

```bash
python scripts/load_dataset.py      # loads and labels the NSL-KDD dataset
python scripts/eda_basic.py         # exploratory data analysis, generates charts
python scripts/preprocessing.py     # feature selection, encoding, scaling
python scripts/model_training.py    # trains and compares models, saves the best one
```

## Project Structure

```
.
├── scripts/
│   ├── load_dataset.py       # NSL-KDD loading and column labelling
│   ├── eda_basic.py          # Exploratory data analysis + chart generation
│   ├── preprocessing.py      # Feature selection, encoding, scaling
│   ├── model_training.py     # Model training, comparison, evaluation
│   └── detector.py           # Live/file-mode intrusion detector
├── requirements.txt
└── README.md
```

> **Note:** the full NSL-KDD dataset, trained model files, and packet captures are not included in this repository to keep it lightweight. The dataset is publicly available — see [NSL-KDD on the Canadian Institute for Cybersecurity site](https://www.unb.ca/cic/datasets/nsl.html). Scripts expect `KDDTrain+.txt` and `KDDTest+.txt` inside a local `dataset/` folder, which is excluded via `.gitignore`.

## Key Engineering Decisions

- **Feature reduction grounded in EDA, not guesswork** — features that were constant or near-constant (e.g. >99% zeros) were dropped, along with features highly correlated with others already in the model (e.g. removing `srv_serror_rate` as a near-duplicate of `serror_rate`), reducing dimensionality without losing predictive signal.
- **Threshold tuned for recall over raw accuracy** — the classification threshold was lowered from the default 0.5 to 0.1, deliberately trading some false positives for fewer missed attacks, which better fits the cost asymmetry in intrusion detection.
- **Class imbalance addressed with SMOTE** — rather than letting the model default to the majority class, synthetic minority oversampling was used during training to better represent attack classes.

## Skills Demonstrated

Network Traffic Analysis · Machine Learning (Random Forest, SMOTE) · Feature Engineering · Python (pandas, scikit-learn, Scapy) · Packet Capture & Parsing · Model Evaluation · Exploratory Data Analysis
