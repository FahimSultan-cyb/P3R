"""
Quick Start Example for P3R-HeadGate Model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import pandas as pd
from models.p3r_headgate import P3RHeadGateModel
from utils.dataset import create_dataloader
from evaluation.metrics import calculate_comprehensive_metrics, print_metrics


def quick_inference_demo():
    """Demonstrate basic inference with P3R-HeadGate"""
    
    # Sample data for demonstration
    sample_data = {
        'func': [
            'def vulnerable_function(user_input): exec(user_input)',
            'def safe_function(user_input): return user_input.strip()',
            'def sql_injection(query): cursor.execute("SELECT * FROM users WHERE id=" + query)',
            'def secure_query(user_id): cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))'
        ],
        'label': [1, 0, 1, 0]  # 1: vulnerable, 0: safe
    }
    
    df = pd.DataFrame(sample_data)
    print("Sample Data:")
    print(df)
    
    # Initialize model
    print("\nInitializing P3R-HeadGate Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    model = P3RHeadGateModel().to(device)
    trainable_params, total_params = model.count_parameters()
    print(f"Model loaded on {device}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.1%})")
    
    # Create dataloader
    dataloader = create_dataloader(df, model.tokenizer, batch_size=2, shuffle=False)
    
    # Run inference
    print("\nRunning inference...")
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            chunks = batch['chunks'].to(device)
            full_code = batch['full_code'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(chunks, full_code, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())
    
    # Display results
    print("\nResults:")
    df['predicted'] = predictions
    df['vulnerability_prob'] = probabilities
    print(df[['func', 'label', 'predicted', 'vulnerability_prob']])
    
    # Calculate metrics
    metrics = calculate_comprehensive_metrics(df['label'].values, 
                                            df['predicted'].values, 
                                            df['vulnerability_prob'].values)
    print_metrics(metrics)


if __name__ == "__main__":
    quick_inference_demo()