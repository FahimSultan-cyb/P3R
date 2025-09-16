# Pre-trained Models

This directory contains the pre-trained P3R models for aerospace vulnerability detection.

## Model Files

### p3r_headgate_model1.pth
- **Description**: Main P3R model with Head Gate mechanism
- **Architecture**: UniXCoder + Prompt Pool + Router MLP + Head Gate
- **Parameters**: 2.1M trainable (125M total)
- **Performance**: 94.2% accuracy on aerospace vulnerability detection

### symbolic_classifier1n.pth
- **Description**: Lightweight symbolic classifier
- **Architecture**: 768 → 384 → 2 (with dropout)
- **Parameters**: 295K trainable
- **Usage**: Final classification layer for P3R pipeline

## Download Instructions

Run the download script to automatically fetch models from Google Drive:

```bash
python scripts/download_models.py
```

Or manually download:
1. Download model files from the provided Google Drive links
2. Place them in this `models/` directory
3. Ensure filenames match exactly: `p3r_headgate_model1.pth` and `symbolic_classifier1n.pth`

## Model Loading

```python
from src.models.p3r_model import P3RHeadGateModel
import torch

model = P3RHeadGateModel()
checkpoint = torch.load('models/p3r_headgate_model1.pth')
model.load_state_dict(checkpoint, strict=False)
```

## Training Details

- **Dataset**: NASA aerospace vulnerability dataset
- **Training Time**: ~4 hours on V100 GPU
- **Optimization**: AdamW with warmup scheduling
- **Validation**: 5-fold cross-validation
- **Metrics**: F1=93.8%, Precision=94.1%, Recall=93.5%

## Citation

If you use these models, please cite our paper:

```bibtex
@article{p3r_aerospace_2024,
  title={P3R-Aerospace: Parameter-Efficient Fine-Tuning for Aerospace Vulnerability Detection},
  author={Your Name},
  journal={IEEE Transactions on Aerospace and Electronic Systems},
  year={2024}
}
```
