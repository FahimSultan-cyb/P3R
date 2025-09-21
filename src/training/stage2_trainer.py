import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from ..data.dataset import CodeDataset, create_collate_fn
from ..models.p3r_stage2_model import P3RStage2Model

class Stage2Trainer:
    def __init__(self, model_name, stage1_classifier_path, num_prompts=4, prompt_length=8, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = P3RStage2Model(
            model_name=model_name,
            stage1_classifier_path=stage1_classifier_path,
            num_prompts=num_prompts,
            prompt_length=prompt_length
        ).to(self.device)
        
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        self.optimizer = None
        self.trainable_params = trainable_params
        
    def train(self, train_df, val_df=None, epochs=10, batch_size=8, lr=2e-5):
        self.optimizer = optim.AdamW(self.trainable_params, lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        collate_fn = create_collate_fn(self.model.tokenizer)
        train_dataset = CodeDataset(train_df, self.model.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
        if val_df is not None:
            val_dataset = CodeDataset(val_df, self.model.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            progress_bar = tqdm(train_loader, desc=f'Stage 2 P3R Epoch {epoch+1}/{epochs}')
            
            for batch in progress_bar:
                chunks = batch['chunks'].to(self.device)
                full_code = batch['full_code'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                self.optimizer.zero_grad()
                
                logits = self.model(chunks, full_code, attention_mask)
                loss = criterion(logits, labels)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
            
            avg_loss = total_loss / len(train_loader)
            accuracy = 100. * correct / total
            print(f"Stage 2 P3R Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            if val_df is not None:
                val_acc = self.validate(val_loader)
                print(f"Validation Accuracy: {val_acc:.2f}%")
    
    def validate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                chunks = batch['chunks'].to(self.device)
                full_code = batch['full_code'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(chunks, full_code, attention_mask)
                _, predicted = torch.max(logits.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100. * correct / total
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Stage 2 P3R model saved to: {path}")
    
    def get_parameter_info(self):
        trainable, total = self.model.count_parameters()
        return trainable, total
