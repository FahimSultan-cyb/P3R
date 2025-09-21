#!/usr/bin/env python3

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from src.preprocessing.neurosymbolic import preprocess_dataset
from src.training.stage1_trainer import Stage1Trainer
from src.training.stage2_trainer import Stage2Trainer

def main():
    parser = argparse.ArgumentParser(description='P3R Two-Stage Training')
    parser.add_argument('--input_data', type=str, required=True, help='Input CSV with func,label columns')
    parser.add_argument('--model_name', type=str, required=True, help='Pretrained model name (e.g., microsoft/unixcoder-base)')
    parser.add_argument('--output_dir', type=str, default='models/', help='Output directory for models')
    parser.add_argument('--stage1_epochs', type=int, default=10, help='Stage 1 training epochs')
    parser.add_argument('--stage2_epochs', type=int, default=5, help='Stage 2 training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test split ratio')
    
    args = parser.parse_args()
    
    print("=== P3R Two-Stage Training Pipeline ===")
    print(f"Input data: {args.input_data}")
    print(f"Pretrained model: {args.model_name}")
    print(f"Output directory: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n=== PREPROCESSING: Neurosymbolic Feature Extraction ===")
    processed_path = os.path.join(args.output_dir, 'preprocessed_data.csv')
    df = preprocess_dataset(args.input_data, processed_path)
    print(f"Generated neurosymbolic features for {len(df)} samples")
    
    print("\n=== DATA SPLITTING ===")
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df['label'])
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    print("\n=== STAGE 1: Training Symbolic Classifier ===")
    stage1_trainer = Stage1Trainer(
        model_name=args.model_name,
        device=args.device
    )
    
    stage1_trainer.train(
        train_df=train_df,
        val_df=val_df,
        epochs=args.stage1_epochs,
        batch_size=args.batch_size
    )
    
    stage1_path = os.path.join(args.output_dir, 'stage1_classifier.pth')
    stage1_trainer.save_classifier(stage1_path)
    
    print("\n=== STAGE 2: Training P3R Components ===")
    stage2_trainer = Stage2Trainer(
        model_name=args.model_name,
        stage1_classifier_path=stage1_path,
        device=args.device
    )
    
    trainable_params, total_params = stage2_trainer.get_parameter_info()
    print(f"Stage 2 Trainable Parameters: {trainable_params:,}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Parameter Efficiency: {trainable_params/total_params:.1%}")
    
    stage2_trainer.train(
        train_df=train_df,
        val_df=val_df,
        epochs=args.stage2_epochs,
        batch_size=args.batch_size
    )
    
    stage2_path = os.path.join(args.output_dir, 'stage2_p3r_model.pth')
    stage2_trainer.save_model(stage2_path)
    
    test_df.to_csv(os.path.join(args.output_dir, 'test_data.csv'), index=False)
    
    print("\n=== TRAINING COMPLETED ===")
    print(f"Stage 1 Classifier: {stage1_path}")
    print(f"Stage 2 P3R Model: {stage2_path}")
    print(f"Test Data: {os.path.join(args.output_dir, 'test_data.csv')}")
    print("\nUse inference script to evaluate the trained model:")
    print(f"python scripts/inference_universal.py --test_data {os.path.join(args.output_dir, 'test_data.csv')} --model_name {args.model_name} --stage1_path {stage1_path} --stage2_path {stage2_path}")

if __name__ == "__main__":
    main()
