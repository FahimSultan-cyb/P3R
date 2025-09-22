#!/usr/bin/env python3

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from src.models.universal_p3r import UniversalP3RModel
from src.data.universal_dataset import Stage1Dataset, Stage2Dataset, create_stage1_collate_fn, create_stage2_collate_fn
from src.training.two_stage_trainer import TwoStageTrainer
from src.preprocessing.neurosymbolic_extractor import preprocess_dataset

def main():
    parser = argparse.ArgumentParser(description='Two-Stage P3R Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--model_name', type=str, default='microsoft/unixcoder-base', 
                       help='HuggingFace model name (e.g., microsoft/codebert-base, microsoft/unixcoder-base)')
    parser.add_argument('--preprocess', action='store_true', help='Run neurosymbolic preprocessing')
    parser.add_argument('--stage1_only', action='store_true', help='Train only stage 1')
    parser.add_argument('--stage2_only', action='store_true', help='Train only stage 2 (requires stage1 classifier)')
    parser.add_argument('--stage1_path', type=str, help='Path to stage 1 classifier for stage 2 training')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using model: {args.model_name}")
    
    if args.preprocess:
        print("Running neurosymbolic preprocessing...")
        preprocessed_path = args.data_path.replace('.csv', '_preprocessed.csv')
        df = preprocess_dataset(args.data_path, preprocessed_path)
        data_path = preprocessed_path
    else:
        data_path = args.data_path
        df = pd.read_csv(data_path)
    
    if 'neuro' not in df.columns and not args.stage2_only:
        raise ValueError("Dataset must have 'neuro' column for stage 1 training. Use --preprocess flag.")
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    model = UniversalP3RModel(
        model_name=args.model_name,
        num_prompts=config['model']['num_prompts'],
        prompt_length=config['model']['prompt_length'],
        stage=1
    ).to(device)
    
    trainer = TwoStageTrainer(
        model=model,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    
    if not args.stage2_only:
        print("\n" + "="*50)
        print("STAGE 1: Training Symbolic Classifier")
        print("="*50)
        
        stage1_train_dataset = Stage1Dataset(train_df, model.tokenizer, config['model']['max_length'])
        stage1_val_dataset = Stage1Dataset(val_df, model.tokenizer, config['model']['max_length'])
        
        stage1_train_loader = DataLoader(
            stage1_train_dataset, 
            batch_size=config['training']['batch_size'],
            shuffle=True, 
            collate_fn=create_stage1_collate_fn()
        )
        
        stage1_val_loader = DataLoader(
            stage1_val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            collate_fn=create_stage1_collate_fn()
        )
        
        stage1_classifier_path = trainer.train_stage1(
            stage1_train_loader, 
            stage1_val_loader,
            epochs=config['training']['stage1_epochs'],
            save_path=config['paths']['model_dir']
        )
        
        print(f"\nStage 1 completed! Classifier saved at: {stage1_classifier_path}")
    
    if not args.stage1_only:
        print("\n" + "="*50)
        print("STAGE 2: Training P3R Components")
        print("="*50)
        
        classifier_path = args.stage1_path if args.stage1_path else os.path.join(config['paths']['model_dir'], 'stage1_classifier.pth')
        
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Stage 1 classifier not found at: {classifier_path}")
        
        stage2_train_dataset = Stage2Dataset(
            train_df, 
            model.tokenizer, 
            config['model']['max_length'],
            config['model']['chunk_size'],
            config['model']['stride']
        )
        
        stage2_val_dataset = Stage2Dataset(
            val_df,
            model.tokenizer,
            config['model']['max_length'],
            config['model']['chunk_size'],
            config['model']['stride']
        )
        
        stage2_train_loader = DataLoader(
            stage2_train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            collate_fn=create_stage2_collate_fn(model.tokenizer)
        )
        
        stage2_val_loader = DataLoader(
            stage2_val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            collate_fn=create_stage2_collate_fn(model.tokenizer)
        )
        
        stage2_model_path = trainer.train_stage2(
            stage2_train_loader,
            stage2_val_loader,
            classifier_path,
            epochs=config['training']['stage2_epochs'],
            save_path=config['paths']['model_dir']
        )
        
        print(f"\nStage 2 completed! P3R model saved at: {stage2_model_path}")
        
        trainable, total = model.count_parameters()
        print(f"\nFinal Model Statistics:")
        print(f"Trainable Parameters: {trainable:,}")
        print(f"Total Parameters: {total:,}")
        print(f"Parameter Efficiency: {trainable/total:.1%}")
    
    print("\nTwo-Stage P3R Training Completed Successfully!")

if __name__ == "__main__":
    main()
