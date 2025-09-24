import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from configs.config import P3RConfig

class PromptPool(nn.Module):
    def __init__(self, num_prompts, prompt_length, embed_dim):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.prompts = nn.Parameter(torch.randn(num_prompts, prompt_length, embed_dim) * 0.02)
        
    def forward(self, weights):
        composite_prompt = torch.zeros(weights.size(0), self.prompt_length, self.embed_dim, device=weights.device)
        for i in range(self.num_prompts):
            composite_prompt += weights[:, i:i+1, None] * self.prompts[i:i+1]
        return composite_prompt

class RouterMLP(nn.Module):
    def __init__(self, embed_dim, num_prompts):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_prompts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, embeddings):
        return self.mlp(embeddings)

class HeadGate(nn.Module):
    def __init__(self, num_layers, num_heads):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.gates = nn.Parameter(torch.ones(num_layers, num_heads))
        
    def apply_gate(self, attention_output, layer_idx):
        if attention_output.dim() == 3:
            batch_size, seq_len, hidden_size = attention_output.shape
            head_size = hidden_size // self.num_heads
            attention_output = attention_output.view(batch_size, seq_len, self.num_heads, head_size)
            gated_output = attention_output * self.gates[layer_idx].view(1, 1, self.num_heads, 1)
            return gated_output.view(batch_size, seq_len, hidden_size)
        return attention_output

class ClassifierMLP(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

class P3RHeadGateModel(nn.Module):
    def __init__(self, model_name=None, config=None):
        super().__init__()
        
        if config is None:
            config = P3RConfig()
        if model_name is not None:
            config.model_name = model_name
            
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.backbone = AutoModel.from_pretrained(config.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.embed_dim = self.backbone.config.hidden_size
        self.num_layers = self.backbone.config.num_hidden_layers
        self.num_heads = self.backbone.config.num_attention_heads
        
        self.prompt_pool = PromptPool(config.num_prompts, config.prompt_length, self.embed_dim)
        self.router = RouterMLP(self.embed_dim, config.num_prompts)
        self.head_gate = HeadGate(self.num_layers, self.num_heads)
        self.classifier = ClassifierMLP(self.embed_dim, config.num_classes, config.dropout)
        
        self._register_hooks()
        
    def _register_hooks(self):
        def create_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 0:
                    gated = self.head_gate.apply_gate(output[0], layer_idx)
                    output = (gated,) + output[1:]
                    return output
                else:
                    return self.head_gate.apply_gate(output, layer_idx)
            return hook
            
        for i, layer in enumerate(self.backbone.encoder.layer):
            layer.attention.self.register_forward_hook(create_hook(i))
    
    def get_chunk_embeddings(self, chunks):
        batch_size, num_chunks, chunk_size = chunks.shape
        chunks_flat = chunks.view(-1, chunk_size)
        
        attention_mask = (chunks_flat != self.tokenizer.pad_token_id).long()
        
        with torch.no_grad():
            chunk_outputs = self.backbone(input_ids=chunks_flat, attention_mask=attention_mask)
            chunk_embeddings = chunk_outputs.last_hidden_state.mean(dim=1)
            
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks, -1)
        return torch.mean(chunk_embeddings, dim=1)
    
    def forward(self, chunks, full_code, attention_mask):
        chunk_embeddings = self.get_chunk_embeddings(chunks)
        prompt_weights = self.router(chunk_embeddings)
        composite_prompt = self.prompt_pool(prompt_weights)
        
        full_code_attention = attention_mask
        prompt_length = composite_prompt.size(1)
        
        if full_code.size(1) + prompt_length > 512:
            truncate_length = 512 - prompt_length
            full_code = full_code[:, :truncate_length]
            full_code_attention = full_code_attention[:, :truncate_length]
        
        inputs_embeds = self.backbone.embeddings.word_embeddings(full_code)
        combined_embeds = torch.cat([composite_prompt, inputs_embeds], dim=1)
        
        extended_attention_mask = torch.ones(attention_mask.size(0), prompt_length, 
                                           device=attention_mask.device, dtype=attention_mask.dtype)
        combined_attention_mask = torch.cat([extended_attention_mask, full_code_attention], dim=1)
        
        outputs = self.backbone(inputs_embeds=combined_embeds, attention_mask=combined_attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        
        return logits
    
    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
