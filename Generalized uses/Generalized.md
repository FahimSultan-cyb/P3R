

# Universal P3R Usage Examples

## Quick Start
```bash
!git clone https://github.com/FahimSultan-cyb/P3R.git
import os, sys
root_path = os.path.join(os.getcwd(), "P3R")
os.chdir(root_path)
!pip install -e .
!python scripts/download_models.py

!pip install -r requirements.txt

```

## Programmatic Usage

### Python API Example
```python
from transformers import AutoTokenizer
import torch
import pandas as pd

from src.models.universal_p3r import UniversalP3RModel
from src.preprocessing.neurosymbolic_extractor import NeurosymbolicFeatureExtractor
from src.training.two_stage_trainer import TwoStageTrainer


extractor = NeurosymbolicFeatureExtractor()
model = UniversalP3RModel("microsoft/codebert-base").to("cuda")

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")


df = pd.read_csv("/content/P3R/data/sample/test.csv")
processed_df = extractor.process_dataset(df)


trainer = TwoStageTrainer(model, "microsoft/codebert-base", device="cuda")
trained_model, classifier_path, model_path = trainer.train_full_pipeline(
    processed_df=processed_df,
    epochs_stage1=10,
    epochs_stage2=10
)
```


## Complete Workflow Examples

### Example 1: Training with CodeBERT
```bash
# Step 1: Preprocess your dataset
!python scripts/preprocess_data.py --input_csv your_dataset_path

```

### Custom Feature Extraction
```python
from src.preprocessing.neurosymbolic_extractor import NeurosymbolicFeatureExtractor

extractor = NeurosymbolicFeatureExtractor()
code_sample = """
int vulnerable_func(char* input) {
    char buffer[256];
    strcpy(buffer, input);
    return 0;
}
"""

features = extractor.extract_all_features(code_sample)
print(f"Extracted {len(features)} features")
```


This completes the universal P3R implementation that works with any CodePTM while maintaining your core P3R architectural approach and two-stage training methodology.

