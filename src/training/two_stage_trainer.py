import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import os

class CompactSymbolicClassifier(nn.Module):
    def __init__(self, embed_dim=768, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

class Stage1Trainer:
    def __init__(self, backbone_model, device, output_dir="models/"):
        self.backbone = backbone_model
        self.device = device
        self.output_dir = output_dir
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.embed_dim = self.backbone.config.hidden_size
        self.symbolic_classifier = CompactSymbolicClassifier(self.embed_dim, 2).to(device)
        
    def train(self, train_loader, val_loader=None, epochs=5, lr=1e-3):
        print("Stage 1: Training CompactSymbolicClassifier on frozen CodePTM embeddings...")
        
        optimizer = optim.Adam(self.symbolic_classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_val_f1 = 0
        
        for epoch in range(epochs):
            self.symbolic_classifier.train()
            self.backbone.eval()
            
            total_loss = 0
            all_preds = []
            all_labels = []
            
            progress_bar = tqdm(train_loader, desc=f'Stage 1 Epoch {epoch+1}/{epochs}')
            
            for batch in progress_bar:
                full_code = batch['full_code'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                with torch.no_grad():
                    backbone_outputs = self.backbone(input_ids=full_code, attention_mask=attention_mask)
                    embeddings = backbone_outputs.last_hidden_state[:, 0, :]
                
                logits = self.symbolic_classifier(embeddings)
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            train_acc = accuracy_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds)
            
            print(f"Stage 1 Epoch {epoch+1} - Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            
            if val_loader is not None:
                val_f1 = self._validate(val_loader)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    self.save_stage1_model()
        
        if val_loader is None:
            self.save_stage1_model()
            
        for param in self.symbolic_classifier.parameters():
            param.requires_grad = False
            
        return self.symbolic_classifier
    
    def _validate(self, val_loader):
        self.symbolic_classifier.eval()
        self.backbone.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                full_code = batch['full_code'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                backbone_outputs = self.backbone(input_ids=full_code, attention_mask=attention_mask)
                embeddings = backbone_outputs.last_hidden_state[:, 0, :]
                
                logits = self.symbolic_classifier(embeddings)
                _, predicted = torch.max(logits, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)
        print(f"Stage 1 Validation - Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        return val_f1
    
    def save_stage1_model(self):
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(self.symbolic_classifier.state_dict(), 
                  f"{self.output_dir}/stage1_symbolic_classifier.pth")

class Stage2Trainer:
    def __init__(self, p3r_model, stage1_classifier, device, output_dir="models/"):
        self.p3r_model = p3r_model
        self.stage1_classifier = stage1_classifier
        self.device = device
        self.output_dir = output_dir
        
        for param in self.p3r_model.backbone.parameters():
            param.requires_grad = False
        for param in self.stage1_classifier.parameters():
            param.requires_grad = False
            
    def train(self, train_loader, val_loader=None, epochs=10, lr=2e-5, weight_decay=0.01):
        print("Stage 2: Training P3R components with frozen backbone and stage1 classifier...")
        
        p3r_params = []
        p3r_params.extend(self.p3r_model.prompt_pool.parameters())
        p3r_params.extend(self.p3r_model.router.parameters())
        p3r_params.extend(self.p3r_model.head_gate.parameters())
        
        optimizer = optim.AdamW(p3r_params, lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        best_val_f1 = 0
        
        for epoch in range(epochs):
            self.p3r_model.train()
            self.stage1_classifier.eval()
            
            total_loss = 0
            all_preds = []
            all_labels = []
            
            progress_bar = tqdm(train_loader, desc=f'Stage 2 Epoch {epoch+1}/{epochs}')
            
            for batch in progress_bar:
                chunks = batch['chunks'].to(self.device)
                full_code = batch['full_code'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                p3r_embeddings = self.p3r_model(chunks, full_code, attention_mask)
                logits = self.stage1_classifier(p3r_embeddings)
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            train_acc = accuracy_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds)
            
            print(f"Stage 2 Epoch {epoch+1} - Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            
            if val_loader is not None:
                val_f1 = self._validate(val_loader)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    self.save_stage2_model()
        
        if val_loader is None:
            self.save_stage2_model()
            
        return self.p3r_model
    
    def _validate(self, val_loader):
        self.p3r_model.eval()
        self.stage1_classifier.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                chunks = batch['chunks'].to(self.device)
                full_code = batch['full_code'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                p3r_embeddings = self.p3r_model(chunks, full_code, attention_mask)
                logits = self.stage1_classifier(p3r_embeddings)
                _, predicted = torch.max(logits, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)
        print(f"Stage 2 Validation - Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        return val_f1
    
    def save_stage2_model(self):
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(self.p3r_model.state_dict(), 
                  f"{self.output_dir}/stage2_p3r_model.pth")

class CorrectedTwoStageTrainer:
    def __init__(self, model_name, device, output_dir="models/"):
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        
    def train_full_pipeline(self, train_loader, val_loader=None, 
                           stage1_epochs=5, stage2_epochs=10, 
                           stage1_lr=1e-3, stage2_lr=2e-5):
        
        from transformers import AutoModel
        from ..models.universal_p3r_model import UniversalP3RModel
        
        backbone = AutoModel.from_pretrained(self.model_name).to(self.device)
        
        stage1_trainer = Stage1Trainer(backbone, self.device, self.output_dir)
        stage1_classifier = stage1_trainer.train(train_loader, val_loader, 
                                               epochs=stage1_epochs, lr=stage1_lr)
        
        p3r_model = UniversalP3RModel(self.model_name).to(self.device)
        p3r_model.classifier = stage1_classifier
        
        stage2_trainer = Stage2Trainer(p3r_model, stage1_classifier, self.device, self.output_dir)
        final_model = stage2_trainer.train(train_loader, val_loader, 
                                         epochs=stage2_epochs, lr=stage2_lr)
        
        print("Two-stage P3R training completed!")
        return final_model, stage1_classifier