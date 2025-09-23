# P3R-HeadGate: Parameter-Efficient Fine-Tuning for Aerospace Vulnerability Detection

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This repository implements P3R-HeadGate, a novel Parameter-Efficient Fine-Tuning (PEFT) approach for aerospace code vulnerability detection. The method combines Prompt Pool learning with Head Gating mechanisms while keeping the backbone UniXCoder model frozen, achieving high performance with minimal trainable parameters.

### Generalization 
We have implemented a generalization approach that can be applied to any model, with the functionality available in the Generalized uses folder.


### Key Features

- **Parameter Efficiency**: Only 0.3% of total parameters are trainable
- **Prompt Pool Learning**: Dynamic prompt selection based on code context
- **Head Gating**: Selective attention head activation for better feature extraction
- **Space Mission Simulation**: Integrated KSP (Kerbal Space Program) mission metrics
- **Comprehensive Evaluation**: 20+ evaluation metrics including aerospace-specific measures

## Quick Start
```bash
!git clone https://github.com/FahimSultan-cyb/P3R.git
import os, sys
root_path = os.path.join(os.getcwd(), "P3R")
os.chdir(root_path)
!pip install -e .
#!python scripts/download_models.py

!pip install -r requirements.txt

```

## Run inference on test data
```bash
!python scripts/inference.py --test_data data/sample/test.csv --output_dir results/

```


## Visualization [Aero-space Metrics]
```bash
python scripts/visualize.py --data results/ksp_mission_data.csv --output outputs/visualizations/dashboard.png
```


## Training 
```bash
!python scripts/train.py --config configs/default.yaml --train_data data/sample/test.csv
```


## Pre-trained Models

Download from Google Drive:
- P3R Model: `models/p3r_headgate_model1.pth`
- Classifier: `models/symbolic_classifier1n.pth`




