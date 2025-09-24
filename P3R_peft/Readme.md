## Installation
```bash
!git clone https://github.com/FahimSultan-cyb/P3R.git
import os, sys
root_path = os.path.join(os.getcwd(), "P3R")
os.chdir(root_path)
!pip install -e .
!python scripts/download_models.py

!pip install -r requirements.txt

peft_root = os.path.join(os.getcwd(), "P3R_peft")
os.chdir(peft_root)
!pip install -e .
!pip install -r requirements.txt

```

## Quick Start

```bash
os.chdir(root_path)
from src.models.p3r_model import P3RHeadGateModel
from src.models.p3r_trainer import P3RTrainer
from transformers import AutoTokenizer
import pandas as pd


model = P3RHeadGateModel("microsoft/codebert-base")
trainer = P3RTrainer(model)

trainer.train("train.csv", epochs=5)

results = trainer.evaluate("test.csv")
```



