#!/usr/bin/env python3

import sys
import os
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from src.preprocessing.neurosymbolic_extractor import NeurosymbolicFeatureExtractor
from src.data.dataset import CodeDataset, create_collate_fn
from src.training.two_stage_trainer import TwoStageTrainer

def main():
    parser = argparse.ArgumentParser(description='Corrected P3R Two-Stage Training')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--model_name', type=str, required=True, help='CodePTM model name')
    parser.add_argument('--output_dir', type=str, default='models/', help='Output directory')
    parser.add_argument('--stage1_epochs', type=int, default=5, help='Stage 1 epochs')
    parser.add_argument('--stage2_epochs', type=int, default=10, help='Stage 2 epochs')
    parser.add_argument('--stage1_lr', type=float, default=1e-3, help='Stage 1 learning rate')
    parser.add_argument('--stage2_lr', type=float, default=2e-5, help='Stage 2 learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Using CodePTM: {args.model_name}")
    
    print("Loading data...")
    df = pd.read_csv(args.train_data)
    print(f"Loaded {len(df)} samples")
    
    if 'neuro' not in df.columns:
        print("Extracting neurosymbolic features for analysis...")
        extractor = NeurosymbolicFeatureExtractor()
        df = extractor.process_dataset(df)
    
    train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=42, stratify=df['label'])
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    print("Creating data loaders...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    collate_fn = create_collate_fn(tokenizer)
    train_dataset = CodeDataset(train_df, tokenizer)
    val_dataset = CodeDataset(val_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print("\n" + "="*50)
    print("CORRECTED P3R TWO-STAGE TRAINING")
    print("="*50)
    
    trainer = TwoStageTrainer(args.model_name, device, args.output_dir)
    
    try:
        final_model, stage1_classifier = trainer.train_full_pipeline(
            train_loader, val_loader,
            stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs,
            stage1_lr=args.stage1_lr, stage2_lr=args.stage2_lr
        )
        
        print("\nTwo-stage P3R training completed successfully!")
        print(f"Models saved to: {args.output_dir}")
        
        trainable, total = final_model.count_parameters()
        print(f"\nFinal P3R Model:")
        print(f"Trainable: {trainable:,} | Total: {total:,} | Efficiency: {trainable/total:.1%}")
        
        stage1_params = sum(p.numel() for p in stage1_classifier.parameters())
        p3r_params = trainable - stage1_params
        print(f"Stage 1 Classifier: {stage1_params:,} (frozen)")
        print(f"P3R Components: {p3r_params:,} (trainable)")
        
        print(f"\nArchitecture Summary:")
        print(f"- Frozen CodePTM Backbone: {total-trainable:,} params")
        print(f"- Frozen CompactSymbolicClassifier: {stage1_params:,} params")
        print(f"- Trainable P3R (Prompt Pool + Router + Head Gate): {p3r_params:,} params")
        print(f"- Total Parameter Efficiency: {trainable/total:.1%}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == "__main__":

    main()

