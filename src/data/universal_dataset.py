import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast

class Stage1Dataset(Dataset):
    def __init__(self, data_df, tokenizer, max_length=512):
        self.data = data_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        neuro_features = str(self.data.iloc[idx]['neuro'])
        label = int(self.data.iloc[idx]['label'])
        
        encoding = self.tokenizer(
            neuro_features,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

class Stage2Dataset(Dataset):
    def __init__(self, data_df, tokenizer, max_length=512, chunk_size=512, stride=256):
        self.data = data_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.stride = stride
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        code = str(self.data.iloc[idx]['func'])
        label = int(self.data.iloc[idx]['label'])
        
        tokens = self.tokenizer.encode(code, add_special_tokens=False, 
                                     max_length=self.max_length, truncation=True)
        
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
            'label': torch.tensor(label, dtype=torch.long),
            'attention_mask': torch.tensor([1 if t != self.tokenizer.pad_token_id else 0 
                                          for t in full_tokens[:self.max_length]], dtype=torch.long)
        }

def create_stage1_collate_fn():
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': labels
        }
    return collate_fn

def create_stage2_collate_fn(tokenizer):
    def collate_fn(batch):
        max_chunks = max(item['chunks'].size(0) for item in batch)
        
        padded_chunks = []
        full_codes = []
        attention_masks = []
        labels = []
        
        for item in batch:
            chunks = item['chunks']
            if chunks.size(0) < max_chunks:
                padding_chunks = torch.full((max_chunks - chunks.size(0), chunks.size(1)), 
                                          tokenizer.pad_token_id, dtype=torch.long)
                chunks = torch.cat([chunks, padding_chunks], dim=0)
            padded_chunks.append(chunks)
            full_codes.append(item['full_code'])
            attention_masks.append(item['attention_mask'])
            labels.append(item['label'])
        
        return {
            'chunks': torch.stack(padded_chunks),
            'full_code': torch.stack(full_codes),
            'attention_mask': torch.stack(attention_masks),
            'label': torch.stack(labels)
        }
    return collate_fn
