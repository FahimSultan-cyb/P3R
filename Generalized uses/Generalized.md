

# Universal P3R Usage Examples

## Quick Start
```bash
!git clone https://github.com/FahimSultan-cyb/P3R-Aerospace.git
!cd P3R-Aerospace
!pip install -e P3R-Aerospace
!python P3R-Aerospace/scripts/download_models.py

# Install dependencies
!pip install -r P3R-Aerospace/requirements.txt

```

## Programmatic Usage

### Python API Example
```python
import torch
from src.models.universal_p3r_model import UniversalP3RModel
from src.preprocessing.neurosymbolic_extractor import NeurosymbolicFeatureExtractor
from src.training.two_stage_trainer import TwoStageTrainer

# Initialize components
extractor = NeurosymbolicFeatureExtractor()
model = UniversalP3RModel("microsoft/codebert-base")

# Process data
processed_df = extractor.process_dataset(raw_df)

# Train model
trainer = TwoStageTrainer("microsoft/codebert-base", device="cuda")
trained_model, classifier, vectorizer = trainer.train_full_pipeline(
    processed_df, train_loader, val_df, val_loader
)
```



## Complete Workflow Examples

### Example 1: Training with CodeBERT
```bash
# Step 1: Preprocess your dataset
!python P3R-Aerospace/scripts/preprocess_dataset.py --input_csv /path/to/your_dataset.csv --output_csv /path/to/preprocessed_dataset.csv

# Step 2: Train with CodeBERT
!python P3R-Aerospace/scripts/train_universal.py --train_data /path/to/preprocessed_dataset.csv --model_name microsoft/codebert-base --output_dir models/codebert/ --stage2_epochs 15 --batch_size 16

# Step 3: Run inference
!python P3R-Aerospace/scripts/inference_universal.py --test_data /path/to/test_data.csv --model_name microsoft/codebert-base --stage2_model models/codebert/stage2_p3r_model.pth --output_dir results/codebert/
```

### Example 2: Training with UniXCoder  
```bash
# Step 1: Preprocess your dataset
!python P3R-Aerospace/scripts/preprocess_dataset.py --input_csv /path/to/your_dataset.csv --output_csv /path/to/preprocessed_dataset.csv

#step 2: Training with Unixcoder
!python P3R-Aerospace/scripts/train_universal.py --train_data preprocessed_data.csv --model_name microsoft/unixcoder-base --output_dir models/unixcoder/ --stage2_epochs 12 --stage2_lr 1e-5 --batch_size 8
```

### Example 3: Training with GraphCodeBERT
```bash
# Step 1: Preprocess your dataset
!python P3R-Aerospace/scripts/preprocess_dataset.py --input_csv /path/to/your_dataset.csv --output_csv /path/to/preprocessed_dataset.csv

#step 2: Training with graphcodebert
!python P3R-Aerospace/scripts/train_universal.py --train_data preprocessed_data.csv --model_name microsoft/graphcodebert-base --output_dir models/graphcodebert/ --val_split 0.15
```

## Model Comparison Workflow

```bash
#!/bin/bash
# Compare multiple CodePTMs

MODELS=(
    "microsoft/codebert-base"
    "microsoft/unixcoder-base" 
    "microsoft/graphcodebert-base"
    "Salesforce/codet5-small"
)

DATA_PATH="preprocessed_data.csv"
TEST_PATH="test_data.csv"

for model in "${MODELS[@]}"; do
    echo "Training with $model..."
    
    model_dir="models/$(basename $model)"
    mkdir -p $model_dir
    
    !python P3R-Aerospace/scripts/train_universal.py \
        --train_data $DATA_PATH \
        --model_name $model \
        --output_dir $model_dir/ \
        --stage2_epochs 10
    
    echo "Running inference with $model..."
    !python P3R-Aerospace/scripts/inference_universal.py \
        --test_data $TEST_PATH \
        --model_name $model \
        --stage2_model $model_dir/stage2_p3r_model.pth \
        --output_dir results/$(basename $model)/
done
```

## Dataset Preprocessing Examples

### Standard Preprocessing
```bash
!python P3R-Aerospace/scripts/preprocess_dataset.py --input_csv raw_data.csv --output_csv processed_data.csv --func_column source_code
```

### Custom Column Names
```bash
!python P3R-Aerospace/scripts/preprocess_dataset.py --input_csv my_dataset.csv --output_csv my_processed.csv --func_column code_content
```

## Configuration Customization



```bash
!python P3R-Aerospace/scripts/train_universal.py --train_data data.csv --config P3R-Aerospace/configs/my_config.yaml
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

## Evaluation and Analysis

### Comprehensive Evaluation
```bash
!python P3R-Aerospace/scripts/inference_universal.py --test_data large_test_set.csv --model_name microsoft/unixcoder-base --stage2_model models/unixcoder/stage2_p3r_model.pth --output_dir detailed_results/ --batch_size 32
```

### Performance Comparison Script
```python
import pandas as pd
import json

models = ["codebert", "unixcoder", "graphcodebert"]
results = {}

for model in models:
    with open(f"results/{model}/inference_summary.json", 'r') as f:
        data = json.load(f)
        results[model] = {
            'accuracy': data['stage2_accuracy'],
            'f1_score': data['stage2_f1'],
            'efficiency': data['parameter_efficiency']
        }

comparison_df = pd.DataFrame(results).T
print(comparison_df)
```

## Production Deployment

### Batch Processing Pipeline
```bash
# Process multiple datasets
for dataset in dataset1.csv dataset2.csv dataset3.csv; do
    echo "Processing $dataset..."
    
    python scripts/preprocess_dataset.py --input_csv $dataset --output_csv processed_$dataset
    
    python scripts/inference_universal.py --test_data processed_$dataset --model_name microsoft/codebert-base --stage2_model models/production_model.pth \
        --output_dir results/$(basename $dataset .csv)/
done
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY P3R-Aerospace/requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python", "scripts/inference_universal.py"]
```

## Troubleshooting Common Issues

### Memory Issues
```bash
# Reduce batch size and sequence length
!python P3R-Aerospace/scripts/train_universal.py --batch_size 4 --model_name microsoft/codebert-base --train_data data.csv
```

### Model Loading Issues
```python
# Verify model compatibility
from transformers import AutoTokenizer, AutoModel

try:
    tokenizer = AutoTokenizer.from_pretrained("your-model")
    model = AutoModel.from_pretrained("your-model")
    print("Model compatible with Universal P3R")
except Exception as e:
    print(f"Model incompatible: {e}")
```

### Feature Extraction Debugging
```python
from src.preprocessing.neurosymbolic_extractor import NeurosymbolicFeatureExtractor

extractor = NeurosymbolicFeatureExtractor()
code = "your_code_sample"

# Test individual extractors
print("Declarations:", extractor.extract_declarations(code))
print("Function calls:", extractor.extract_function_calls(code))
print("Vulnerability patterns:", extractor.extract_vulnerability_patterns(code))
```

This completes the universal P3R implementation that works with any CodePTM while maintaining your core P3R architectural approach and two-stage training methodology.

