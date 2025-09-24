import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CodeDataset(Dataset):
    def __init__(self, csv_file, tokenizer, config):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.config = config
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        code = str(self.data.iloc[idx][self.config.code_col])
        label = int(self.data.iloc[idx][self.config.label_col])
        
        tokens = self.tokenizer.encode(code, add_special_tokens=False, 
                                     max_length=self.config.max_length, truncation=True)
        
        chunks = []
        if len(tokens) <= self.config.chunk_size:
            chunk_padded = tokens + [self.tokenizer.pad_token_id] * (self.config.chunk_size - len(tokens))
            chunks.append(chunk_padded)
        else:
            for i in range(0, len(tokens), self.config.stride):
                chunk = tokens[i:i + self.config.chunk_size]
                if len(chunk) < 50:
                    break
                if len(chunk) < self.config.chunk_size:
                    chunk = chunk + [self.tokenizer.pad_token_id] * (self.config.chunk_size - len(chunk))
                chunks.append(chunk)
        
        full_tokens = tokens + [self.tokenizer.pad_token_id] * (self.config.max_length - len(tokens))
        
        return {
            'chunks': torch.tensor(chunks, dtype=torch.long),
            'full_code': torch.tensor(full_tokens[:self.config.max_length], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'attention_mask': torch.tensor([1 if t != self.tokenizer.pad_token_id else 0 
                                          for t in full_tokens[:self.config.max_length]], dtype=torch.long)
        }

def create_dataloader(csv_file, tokenizer, config, shuffle=False):
    dataset = CodeDataset(csv_file, tokenizer, config)
    
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
    
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, collate_fn=collate_fn)
