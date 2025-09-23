# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import os
# from tqdm import tqdm
# from sklearn.metrics import accuracy_score, f1_score

# class TwoStageTrainer:
#     def __init__(self, model, device='cuda', learning_rate=2e-5, weight_decay=0.01):
#         self.model = model
#         self.device = device
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.criterion = nn.CrossEntropyLoss()
        
#     def train_stage1(self, train_loader, val_loader, epochs=10, save_path='models/'):
#         print("Starting Stage 1 Training: Neurosymbolic Features → Symbolic Classifier")
        
#         self.model.stage = 1
#         optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
#         best_f1 = 0.0
#         os.makedirs(save_path, exist_ok=True)
        
#         for epoch in range(epochs):
#             self.model.train()
#             total_loss = 0
#             all_preds = []
#             all_labels = []
            
#             progress_bar = tqdm(train_loader, desc=f'Stage1 Epoch {epoch+1}/{epochs}')
#             for batch in progress_bar:
#                 input_ids = batch['input_ids'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 labels = batch['label'].to(self.device)
                
#                 optimizer.zero_grad()
#                 logits = self.model(input_ids, attention_mask)
#                 loss = self.criterion(logits, labels)
                
#                 loss.backward()
#                 optimizer.step()
                
#                 total_loss += loss.item()
#                 _, predicted = torch.max(logits, 1)
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
                
#                 progress_bar.set_postfix({
#                     'loss': f'{loss.item():.4f}',
#                     'acc': f'{accuracy_score(all_labels, all_preds):.3f}'
#                 })
            
#             avg_loss = total_loss / len(train_loader)
#             train_acc = accuracy_score(all_labels, all_preds)
#             train_f1 = f1_score(all_labels, all_preds)
            
#             val_acc, val_f1, val_loss = self.evaluate_stage1(val_loader)
            
#             print(f'Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f}')
#             print(f'Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}, Val Loss: {val_loss:.4f}')
            
#             if val_f1 > best_f1:
#                 best_f1 = val_f1
#                 classifier_path = os.path.join(save_path, 'stage1_classifier.pth')
#                 torch.save(self.model.symbolic_classifier.state_dict(), classifier_path)
#                 print(f'Stage 1 classifier saved: {classifier_path}')
        
#         return classifier_path
    
#     def train_stage2(self, train_loader, val_loader, classifier_path, epochs=10, save_path='models/'):
#         print("Starting Stage 2 Training: Raw Code → P3R → Frozen Classifier")
        
#         self.model.stage = 2
#         self.model.load_stage1_classifier(classifier_path)
        
#         optimizer = optim.AdamW([p for p in self.model.parameters() if p.requires_grad], 
#                               lr=self.learning_rate, weight_decay=self.weight_decay)
        
#         best_f1 = 0.0
#         os.makedirs(save_path, exist_ok=True)
        
#         for epoch in range(epochs):
#             self.model.train()
#             total_loss = 0
#             all_preds = []
#             all_labels = []
            
#             progress_bar = tqdm(train_loader, desc=f'Stage2 Epoch {epoch+1}/{epochs}')
#             for batch in progress_bar:
#                 chunks = batch['chunks'].to(self.device)
#                 full_code = batch['full_code'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 labels = batch['label'].to(self.device)
                
#                 optimizer.zero_grad()
#                 logits = self.model(chunks, full_code, attention_mask)
#                 loss = self.criterion(logits, labels)
                
#                 loss.backward()
#                 optimizer.step()
                
#                 total_loss += loss.item()
#                 _, predicted = torch.max(logits, 1)
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
                
#                 progress_bar.set_postfix({
#                     'loss': f'{loss.item():.4f}',
#                     'acc': f'{accuracy_score(all_labels, all_preds):.3f}'
#                 })
            
#             avg_loss = total_loss / len(train_loader)
#             train_acc = accuracy_score(all_labels, all_preds)
#             train_f1 = f1_score(all_labels, all_preds)
            
#             val_acc, val_f1, val_loss = self.evaluate_stage2(val_loader)
            
#             print(f'Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f}')
#             print(f'Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}, Val Loss: {val_loss:.4f}')
            
#             if val_f1 > best_f1:
#                 best_f1 = val_f1
#                 model_path = os.path.join(save_path, 'stage2_p3r_model.pth')
#                 torch.save(self.model.state_dict(), model_path)
#                 print(f'Stage 2 P3R model saved: {model_path}')
        
#         return model_path
    
#     def evaluate_stage1(self, val_loader):
#         self.model.eval()
#         total_loss = 0
#         all_preds = []
#         all_labels = []
        
#         with torch.no_grad():
#             for batch in val_loader:
#                 input_ids = batch['input_ids'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 labels = batch['label'].to(self.device)
                
#                 logits = self.model(input_ids, attention_mask)
#                 loss = self.criterion(logits, labels)
                
#                 total_loss += loss.item()
#                 _, predicted = torch.max(logits, 1)
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
        
#         avg_loss = total_loss / len(val_loader)
#         accuracy = accuracy_score(all_labels, all_preds)
#         f1 = f1_score(all_labels, all_preds)
        
#         return accuracy, f1, avg_loss
    
#     def evaluate_stage2(self, val_loader):
#         self.model.eval()
#         total_loss = 0
#         all_preds = []
#         all_labels = []
        
#         with torch.no_grad():
#             for batch in val_loader:
#                 chunks = batch['chunks'].to(self.device)
#                 full_code = batch['full_code'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 labels = batch['label'].to(self.device)
                
#                 logits = self.model(chunks, full_code, attention_mask)
#                 loss = self.criterion(logits, labels)
                
#                 total_loss += loss.item()
#                 _, predicted = torch.max(logits, 1)
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
        
#         avg_loss = total_loss / len(val_loader)
#         accuracy = accuracy_score(all_labels, all_preds)
#         f1 = f1_score(all_labels, all_preds)
        
#         return accuracy, f1, avg_loss

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import os
# from tqdm import tqdm
# from sklearn.metrics import accuracy_score, f1_score

# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer

# class CodeDataset(Dataset):
#     def __init__(self, csv_file, tokenizer, max_length=512, chunk_size=512, stride=256, code_col='func', label_col='target'):
#         self.data = pd.read_csv(csv_file)
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.chunk_size = chunk_size
#         self.stride = stride
#         self.code_col = code_col
#         self.label_col = label_col
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         code = str(self.data.iloc[idx][self.code_col])
#         label = int(self.data.iloc[idx][self.label_col])
        
#         tokens = self.tokenizer.encode(code, add_special_tokens=False, max_length=self.max_length, truncation=True)
        
#         chunks = []
#         if len(tokens) <= self.chunk_size:
#             chunk_padded = tokens + [self.tokenizer.pad_token_id] * (self.chunk_size - len(tokens))
#             chunks.append(chunk_padded)
#         else:
#             for i in range(0, len(tokens), self.stride):
#                 chunk = tokens[i:i + self.chunk_size]
#                 if len(chunk) < 50:
#                     break
#                 if len(chunk) < self.chunk_size:
#                     chunk = chunk + [self.tokenizer.pad_token_id] * (self.chunk_size - len(chunk))
#                 chunks.append(chunk)
        
#         full_tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
#         return {
#             'chunks': torch.tensor(chunks, dtype=torch.long),
#             'full_code': torch.tensor(full_tokens[:self.max_length], dtype=torch.long),
#             'label': torch.tensor(label, dtype=torch.long),
#             'attention_mask': torch.tensor([1 if t != self.tokenizer.pad_token_id else 0 for t in full_tokens[:self.max_length]], dtype=torch.long)
#         }

# class TwoStageTrainer:
#     def __init__(self, model, device='cuda', learning_rate=2e-5, weight_decay=0.01):
#         self.model = model
#         self.device = device
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.criterion = nn.CrossEntropyLoss()
        
#     def train_stage1(self, train_loader, val_loader, epochs=10, save_path='models/'):
#         print("Starting Stage 1 Training: Neurosymbolic Features → Symbolic Classifier")
        
#         self.model.set_stage(1)
        
#         trainable_params = [p for p in self.model.parameters() if p.requires_grad]
#         print(f"Stage 1 trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
#         if not trainable_params:
#             raise ValueError("No trainable parameters found for Stage 1")
        
#         optimizer = optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        
#         best_f1 = 0.0
#         os.makedirs(save_path, exist_ok=True)
#         classifier_path = os.path.join(save_path, 'stage1_classifier.pth')  # define before loop
       
        
#         for epoch in range(epochs):
#             self.model.train()
#             total_loss = 0
#             all_preds = []
#             all_labels = []
            
#             progress_bar = tqdm(train_loader, desc=f'Stage1 Epoch {epoch+1}/{epochs}')
#             for batch in progress_bar:
#                 input_ids = batch['input_ids'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 labels = batch['label'].to(self.device)
                
#                 optimizer.zero_grad()
#                 logits = self.model(input_ids, attention_mask)
#                 loss = self.criterion(logits, labels)
                
#                 loss.backward()
#                 optimizer.step()
                
#                 total_loss += loss.item()
#                 _, predicted = torch.max(logits, 1)
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
                
#                 progress_bar.set_postfix({
#                     'loss': f'{loss.item():.4f}',
#                     'acc': f'{accuracy_score(all_labels, all_preds):.3f}'
#                 })
            
#             avg_loss = total_loss / len(train_loader)
#             train_acc = accuracy_score(all_labels, all_preds)
#             train_f1 = f1_score(all_labels, all_preds, zero_division=0)
            
#             val_acc, val_f1, val_loss = self.evaluate_stage1(val_loader)
            
#             print(f'Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f}')
#             print(f'Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}, Val Loss: {val_loss:.4f}')
            
#             if val_f1 > best_f1:
#                 best_f1 = val_f1
#                 classifier_path = os.path.join(save_path, 'stage1_classifier.pth')
#                 torch.save(self.model.symbolic_classifier.state_dict(), classifier_path)
#                 print(f'Stage 1 classifier saved: {classifier_path}')
        
#         return classifier_path
    
#     def train_stage2(self, train_loader, val_loader, classifier_path, epochs=10, save_path='models/'):
#         print("Starting Stage 2 Training: Raw Code → P3R → Frozen Classifier")
        
#         self.model.set_stage(2)
#         self.model.load_stage1_classifier(classifier_path)
        
#         trainable_params = [p for p in self.model.parameters() if p.requires_grad]
#         print(f"Stage 2 trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
#         if not trainable_params:
#             raise ValueError("No trainable parameters found for Stage 2")
        
#         optimizer = optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        
#         best_f1 = 0.0
#         os.makedirs(save_path, exist_ok=True)
#         model_path = os.path.join(save_path, 'stage2_p3r_model.pth')  # define here
        
#         for epoch in range(epochs):
#             self.model.train()
#             total_loss = 0
#             all_preds = []
#             all_labels = []
            
#             progress_bar = tqdm(train_loader, desc=f'Stage2 Epoch {epoch+1}/{epochs}')
#             for batch in progress_bar:
#                 chunks = batch['chunks'].to(self.device)
#                 full_code = batch['full_code'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 labels = batch['label'].to(self.device)
                
#                 optimizer.zero_grad()
#                 logits = self.model(chunks, full_code, attention_mask)
#                 loss = self.criterion(logits, labels)
                
#                 loss.backward()
#                 optimizer.step()
                
#                 total_loss += loss.item()
#                 _, predicted = torch.max(logits, 1)
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
                
#                 progress_bar.set_postfix({
#                     'loss': f'{loss.item():.4f}',
#                     'acc': f'{accuracy_score(all_labels, all_preds):.3f}'
#                 })
            
#             avg_loss = total_loss / len(train_loader)
#             train_acc = accuracy_score(all_labels, all_preds)
#             train_f1 = f1_score(all_labels, all_preds, zero_division=0)
            
#             val_acc, val_f1, val_loss = self.evaluate_stage2(val_loader)
            
#             print(f'Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f}')
#             print(f'Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}, Val Loss: {val_loss:.4f}')
            
#             if val_f1 > best_f1:
#                 best_f1 = val_f1
#                 model_path = os.path.join(save_path, 'stage2_p3r_model.pth')
#                 torch.save(self.model.state_dict(), model_path)
#                 print(f'Stage 2 P3R model saved: {model_path}')
        
#         return model_path
    
#     def evaluate_stage1(self, val_loader):
#         self.model.eval()
#         total_loss = 0
#         all_preds = []
#         all_labels = []
        
#         with torch.no_grad():
#             for batch in val_loader:
#                 input_ids = batch['input_ids'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 labels = batch['label'].to(self.device)
                
#                 logits = self.model(input_ids, attention_mask)
#                 loss = self.criterion(logits, labels)
                
#                 total_loss += loss.item()
#                 _, predicted = torch.max(logits, 1)
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
        
#         avg_loss = total_loss / len(val_loader)
#         accuracy = accuracy_score(all_labels, all_preds)
#         f1 = f1_score(all_labels, all_preds, zero_division=0)
        
#         return accuracy, f1, avg_loss
    
#     def evaluate_stage2(self, val_loader):
#         self.model.eval()
#         total_loss = 0
#         all_preds = []
#         all_labels = []
        
#         with torch.no_grad():
#             for batch in val_loader:
#                 chunks = batch['chunks'].to(self.device)
#                 full_code = batch['full_code'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 labels = batch['label'].to(self.device)
                
#                 logits = self.model(chunks, full_code, attention_mask)
#                 loss = self.criterion(logits, labels)
                
#                 total_loss += loss.item()
#                 _, predicted = torch.max(logits, 1)
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
        
#         avg_loss = total_loss / len(val_loader)
#         accuracy = accuracy_score(all_labels, all_preds)
#         f1 = f1_score(all_labels, all_preds, zero_division=0)
        
#         return accuracy, f1, avg_loss

#     def train_full_pipeline(self, train_csv, val_csv, epochs_stage1=10, epochs_stage2=10, save_path="models/",
#                             batch_size=16, code_col="func", label_col="target"):
   
#         train_dataset = CodeDataset(train_csv, tokenizer=self.tokenizer, code_col=code_col, label_col=label_col)
#         val_dataset = CodeDataset(val_csv, tokenizer=self.tokenizer, code_col=code_col, label_col=label_col)

#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


        # classifier_path = self.train_stage1(train_loader, val_loader, epochs=epochs_stage1, save_path=save_path)


        # model_path = self.train_stage2(train_loader, val_loader, classifier_path, epochs=epochs_stage2, save_path=save_path)

        # return self.model, classifier_path, model_path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel

class ProcessedDataset(Dataset):
    def __init__(self, processed_df, stage='stage1'):
        self.data = processed_df
        self.stage = stage
        self._validate_processed_columns()
        
    def _validate_processed_columns(self):
        if self.stage == 'stage1':
            required_cols = ['neuro', 'label']
        else:
            required_cols = ['func', 'label']
            
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            available_cols = self.data.columns.tolist()
            raise ValueError(f"Missing required columns in processed dataframe: {missing_cols}. Available columns: {available_cols}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        if self.stage == 'stage1':
            return {
                'neuro_features': row['neuro'],
                'label': torch.tensor(row['label'], dtype=torch.long)
            }
        else:
            return {
                'func_code': row['func'],
                'label': torch.tensor(row['label'], dtype=torch.long)
            }

class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512, chunk_size=512, stride=256):
        self.data = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.stride = stride
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        code = str(self.data.iloc[idx]['func'])
        label = int(self.data.iloc[idx]['label'])
        
        tokens = self.tokenizer.encode(code, add_special_tokens=False, max_length=self.max_length, truncation=True)
        
        chunks = []
        if len(tokens) <= self.chunk_size:
            chunk_padded = tokens + [self.tokenizer.pad_token_id] * (self.chunk_size - len(tokens))
            chunks.append(chunk_padded)
        else:
            for i in range(0, len(tokens), self.stride):
                chunk = tokens[i:i + self.chunk_size]
                if len(chunk) < 50:
                    break
                if len(chunk) < self.chunk_size:
                    chunk = chunk + [self.tokenizer.pad_token_id] * (self.chunk_size - len(chunk))
                chunks.append(chunk)
        
        full_tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
        return {
            'chunks': torch.tensor(chunks, dtype=torch.long),
            'full_code': torch.tensor(full_tokens[:self.max_length], dtype=torch.long),
            'input_ids': torch.tensor(full_tokens[:self.max_length], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'attention_mask': torch.tensor([1 if t != self.tokenizer.pad_token_id else 0 for t in full_tokens[:self.max_length]], dtype=torch.long)
        }

class TwoStageTrainer:
    def __init__(self, model, backbone_model_name, device='cuda', learning_rate=2e-5, weight_decay=0.01, batch_size=8):
        self.model = model
        self.backbone_model_name = backbone_model_name
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.backbone_tokenizer = AutoTokenizer.from_pretrained(backbone_model_name)
        self.backbone_encoder = AutoModel.from_pretrained(backbone_model_name).to(device)
        for param in self.backbone_encoder.parameters():
            param.requires_grad = False
        
    def vectorize_neuro_features(self, neuro_text_list):
        vectorized_features = []
        for neuro_text in neuro_text_list:
            inputs = self.backbone_tokenizer(str(neuro_text), return_tensors='pt', max_length=512, truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.backbone_encoder(**inputs)
                vector = outputs.last_hidden_state.mean(dim=1).squeeze()
                vectorized_features.append(vector)
        return torch.stack(vectorized_features)
        
    def create_stage1_dataloaders(self, processed_df, test_size=0.2, random_state=42):
        train_df, val_df = train_test_split(processed_df, test_size=test_size, random_state=random_state, stratify=processed_df['label'])
        train_dataset = ProcessedDataset(train_df, stage='stage1')
        val_dataset = ProcessedDataset(val_df, stage='stage1')
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.stage1_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.stage1_collate_fn)
        return train_loader, val_loader
        
    def stage1_collate_fn(self, batch):
        neuro_features = [item['neuro_features'] for item in batch]
        labels = [item['label'] for item in batch]
        vectorized_features = self.vectorize_neuro_features(neuro_features)
        labels_tensor = torch.stack(labels)
        return {'embeddings': vectorized_features, 'label': labels_tensor}
        
    def train_stage1(self, train_loader, val_loader, epochs=10, save_path='models/'):
        self.model.set_stage(1)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found for Stage 1")
        optimizer = optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        best_f1 = 0.0
        os.makedirs(save_path, exist_ok=True)
        classifier_path = os.path.join(save_path, 'stage1_classifier.pth')
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            all_preds = []
            all_labels = []
            progress_bar = tqdm(train_loader, desc=f'Stage1 Epoch {epoch+1}/{epochs}')
            for batch in progress_bar:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['label'].to(self.device)
                optimizer.zero_grad()
                logits = self.model.forward_stage1(embeddings)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy_score(all_labels, all_preds):.3f}'})
            avg_loss = total_loss / len(train_loader)
            train_acc = accuracy_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds, zero_division=0)
            val_acc, val_f1, val_loss = self.evaluate_stage1(val_loader)
            print(f'Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.3f}, Train F1: {train_f1:.3f}')
            print(f'Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}, Val Loss: {val_loss:.4f}')
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.symbolic_classifier.state_dict(), classifier_path)
                print(f'Stage 1 classifier saved: {classifier_path}')
        return classifier_path

    def create_stage2_dataloaders(self, processed_df, test_size=0.2, random_state=42):
            train_df, val_df = train_test_split(
                processed_df,
                test_size=test_size,
                random_state=random_state,
                stratify=processed_df['label']
            )

            train_dataset = CodeDataset(train_df, self.backbone_tokenizer)
            val_dataset = CodeDataset(val_df, self.backbone_tokenizer)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

            return train_loader, val_loader



    
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
        model_path = os.path.join(save_path, 'stage2_p3r_model.pth')
        
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
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['label'].to(self.device)
                logits = self.model.forward_stage1(embeddings)
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

    def train_full_pipeline(self, processed_df, epochs_stage1=10, epochs_stage2=10, save_path="models/"):
        train_loader_stage1, val_loader_stage1 = self.create_stage1_dataloaders(processed_df)
        classifier_path = self.train_stage1(train_loader_stage1, val_loader_stage1, epochs=epochs_stage1, save_path=save_path)
        
        train_loader_stage2, val_loader_stage2 = self.create_stage2_dataloaders(processed_df)
        model_path = self.train_stage2(train_loader_stage2, val_loader_stage2, classifier_path, epochs=epochs_stage2, save_path=save_path)
        
        return self.model, classifier_path, model_path


