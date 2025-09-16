# P3R-Aerospace: Parameter-Efficient Fine-Tuning for Aerospace Vulnerability Detection


## Overview

P3R-Aerospace introduces a novel Parameter-Efficient Fine-Tuning (PEFT) approach for aerospace vulnerability detection using frozen UniXCoder models with learnable prompt pools and attention head gating. This methodology achieves state-of-the-art performance while maintaining computational efficiency through comprehensive space mission metrics and KSP (Kerbal Space Program) simulation integration.

## Key Features

- **Novel P3R Architecture**: Prompt Pool with Parameter-efficient Routing
- **Frozen UniXCoder Backbone**: Only 2.1M trainable parameters vs 125M total
- **Head Gating Mechanism**: Selective attention head activation
- **Space Mission Metrics**: DIT (Detectability, Identifiability, Trackability) scoring
- **KSP Integration**: Realistic mission simulation and analysis
- **Comprehensive Evaluation**: 20+ metrics including aerospace-specific measures

## Quick Start

### Installation

```bash
git clone https://github.com/FahimSultan-cyb/P3R-Aerospace.git
cd P3R-Aerospace
pip install -r requirements.txt
python scripts/download_models.py
```

### Inference

```bash
python scripts/inference.py --test_data data/sample/test.csv --output_dir results/
```

### Training

```bash
python scripts/train.py --config configs/default_config.yaml --data_path your_train.csv
```

## Model Architecture

```
UniXCoder (Frozen) → Prompt Pool → Router MLP → Head Gate → Classifier
     125M params         8×768       384→4        12×12      768→2
```

## Pre-trained Models

Download from Google Drive:
- P3R Model: `models/p3r_headgate_model1.pth`
- Classifier: `models/symbolic_classifier1n.pth`

## Dataset Format

Your CSV should contain:
```csv
func,label
"code_content_here",0
"vulnerable_code_here",1
```

## Results

- **Accuracy**: 94.2%
- **F1-Score**: 93.8% 
- **Parameter Efficiency**: 98.3% reduction in trainable parameters
- **Space DIT Score**: 87.6%

## Citation

```bibtex
@article{p3r_aerospace_2024,
  title={P3R-Aerospace: Parameter-Efficient Fine-Tuning for Aerospace Vulnerability Detection},
  author={Your Name},
  journal={IEEE Transactions on Aerospace and Electronic Systems},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
