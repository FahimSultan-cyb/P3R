import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.p3r_model import P3RHeadGateModel
from src.data.dataset import CodeDataset, create_collate_fn

def train_model(config):
    device = torch.device(config['device'] if config['device'] != 'auto' else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading training data...")
    train_df = pd.read_csv(config['train_data'])
    
    print(f"Initializing P3R model...")
    model = P3RHeadGateModel(
        model_name=config['model']['name'],
        num_prompts=config['model']['num_prompts'],
        prompt_length=config['model']['prompt_length']
    ).to(device)
    
    trainable_params, total_params = model.count_parameters()
    print(f"Trainable: {trainable_params:,} | Total: {total_params:,}")
    
    collate_fn = create_collate_fn(model.tokenizer)
    train_dataset = CodeDataset(train_df, model.tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                             shuffle=True, collate_fn=collate_fn)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting training for {config['training']['epochs']} epochs...")
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]}')
        
        for batch in progress_bar:
            chunks = batch['chunks'].to(device)
            full_code = batch['full_code'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(chunks, full_code, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    model_path = os.path.join(config['paths']['model_dir'], "p3r_trained_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training CSV')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['train_data'] = args.train_data
    
    train_model(config)

if __name__ == "__main__":
    main()