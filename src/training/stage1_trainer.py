import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
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

class Stage1Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.data = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        neuro_features = str(self.data.iloc[idx]['neuro'])
        label = int(self.data.iloc[idx]['label'])
        
        encoding = self.tokenizer(
            neuro_features,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class Stage1Trainer:
    def __init__(self, model_name, embed_dim=768, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.embed_dim = self.backbone.config.hidden_size
        self.classifier = CompactSymbolicClassifier(self.embed_dim, 2).to(self.device)
        
    def get_embeddings(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings
    
    def train(self, train_df, val_df=None, epochs=10, batch_size=16, lr=2e-5):
        train_dataset = Stage1Dataset(train_df, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_df is not None:
            val_dataset = Stage1Dataset(val_df, self.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = optim.AdamW(self.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.classifier.train()
            total_loss = 0
            correct = 0
            total = 0
            
            progress_bar = tqdm(train_loader, desc=f'Stage 1 Epoch {epoch+1}/{epochs}')
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                embeddings = self.get_embeddings(input_ids, attention_mask)
                
                optimizer.zero_grad()
                logits = self.classifier(embeddings)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
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
            print(f"Stage 1 Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            if val_df is not None:
                val_acc = self.validate(val_loader)
                print(f"Validation Accuracy: {val_acc:.2f}%")
    
    def validate(self, val_loader):
        self.classifier.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                embeddings = self.get_embeddings(input_ids, attention_mask)
                logits = self.classifier(embeddings)
                _, predicted = torch.max(logits.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100. * correct / total
    
    def save_classifier(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.classifier.state_dict(), path)
        print(f"Stage 1 classifier saved to: {path}")
    
    def load_classifier(self, path):
        self.classifier.load_state_dict(torch.load(path, map_location=self.device))
        for param in self.classifier.parameters():
            param.requires_grad = False
        print(f"Stage 1 classifier loaded and frozen from: {path}")
