import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

class TwoStageTrainer:
    def __init__(self, model, device='cuda', learning_rate=2e-5, weight_decay=0.01):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()
        
    def train_stage1(self, train_loader, val_loader, epochs=10, save_path='models/'):
        print("Starting Stage 1 Training: Neurosymbolic Features → Symbolic Classifier")
        
        self.model.set_stage(1)
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Stage 1 trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        if not trainable_params:
            raise ValueError("No trainable parameters found for Stage 1")
        
        optimizer = optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        
        best_f1 = 0.0
        os.makedirs(save_path, exist_ok=True)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            all_preds = []
            all_labels = []
            
            progress_bar = tqdm(train_loader, desc=f'Stage1 Epoch {epoch+1}/{epochs}')
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy_score(all_labels, all_preds):.3f}'
                })
            
            avg_loss = total_loss / len(train_loader)
            train_acc = accuracy_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds, zero_division=0)
            
            val_acc, val_f1, val_loss = self.evaluate_stage1(val_loader)
            
            print(f'Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f}')
            print(f'Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}, Val Loss: {val_loss:.4f}')
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                classifier_path = os.path.join(save_path, 'stage1_classifier.pth')
                torch.save(self.model.symbolic_classifier.state_dict(), classifier_path)
                print(f'Stage 1 classifier saved: {classifier_path}')
        
        return classifier_path
    
    def train_stage2(self, train_loader, val_loader, classifier_path, epochs=10, save_path='models/'):
        print("Starting Stage 2 Training: Raw Code → P3R → Frozen Classifier")
        
        self.model.set_stage(2)
        self.model.load_stage1_classifier(classifier_path)
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Stage 2 trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        if not trainable_params:
            raise ValueError("No trainable parameters found for Stage 2")
        
        optimizer = optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        
        best_f1 = 0.0
        os.makedirs(save_path, exist_ok=True)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            all_preds = []
            all_labels = []
            
            progress_bar = tqdm(train_loader, desc=f'Stage2 Epoch {epoch+1}/{epochs}')
            for batch in progress_bar:
                chunks = batch['chunks'].to(self.device)
                full_code = batch['full_code'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(chunks, full_code, attention_mask)
                loss = self.criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy_score(all_labels, all_preds):.3f}'
                })
            
            avg_loss = total_loss / len(train_loader)
            train_acc = accuracy_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds, zero_division=0)
            
            val_acc, val_f1, val_loss = self.evaluate_stage2(val_loader)
            
            print(f'Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f}')
            print(f'Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}, Val Loss: {val_loss:.4f}')
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                model_path = os.path.join(save_path, 'stage2_p3r_model.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f'Stage 2 P3R model saved: {model_path}')
        
        return model_path
    
    def evaluate_stage1(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        return accuracy, f1, avg_loss
    
    def evaluate_stage2(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                chunks = batch['chunks'].to(self.device)
                full_code = batch['full_code'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(chunks, full_code, attention_mask)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        return accuracy, f1, avg_loss
