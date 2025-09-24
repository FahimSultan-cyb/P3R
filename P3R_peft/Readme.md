## Installation
```bash
!git clone https://github.com/FahimSultan-cyb/P3R.git
import os, sys
root_path = os.path.join(os.getcwd(), "P3R")
os.chdir(root_path)
!pip install -e .
!python scripts/download_models.py

!pip install -r requirements.txt

```

## Quick Start
```bash
from p3r_model import P3RHeadGateModel
from p3r_trainer import P3RTrainer
from transformers import AutoTokenizer
import pandas as pd

# Initialize with any CodeLLM
model = P3RHeadGateModel("microsoft/codebert-base")
trainer = P3RTrainer(model)

# Train
trainer.train("train.csv", epochs=5)

# Evaluate
results = trainer.evaluate("test.csv")
```
