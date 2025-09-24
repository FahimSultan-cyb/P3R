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
from src.models.p3r_model import P3RHeadGateModel
from src.models.p3r_trainer import P3RTrainer
from transformers import AutoTokenizer
import pandas as pd


model = P3RHeadGateModel("microsoft/codebert-base")
trainer = P3RTrainer(model)

trainer.train("train.csv", epochs=5)

results = trainer.evaluate("test.csv")
```


## Parameter Configuration
```bash
from configs import P3RConfig

config = P3RConfig(
    model_name="Salesforce/codet5-base",
    num_prompts=8,
    prompt_length=12,
    max_length=1024,
    chunk_size=256,
    learning_rate=5e-5,
    code_col='source_code',
    label_col='is_vulnerable'
)

```
