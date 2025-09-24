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
        if attention_output is None or layer_idx >= self.num_layers:
            return attention_output
        if attention_output.dim() == 3:
            batch_size, seq_len, hidden_size = attention_output.shape
            head_size = hidden_size // self.num_heads
            if head_size * self.num_heads == hidden_size:
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
            
        self._extract_model_config()
        
        self.prompt_pool = PromptPool(config.num_prompts, config.prompt_length, self.embed_dim)
        self.router = RouterMLP(self.embed_dim, config.num_prompts)
        self.head_gate = HeadGate(self.num_layers, self.num_heads)
        self.classifier = ClassifierMLP(self.embed_dim, config.num_classes, config.dropout)
        
        self._register_hooks()
        
    def _extract_model_config(self):
        if hasattr(self.backbone, 'config'):
            self.embed_dim = getattr(self.backbone.config, 'hidden_size', 
                                   getattr(self.backbone.config, 'd_model', 768))
            self.num_layers = getattr(self.backbone.config, 'num_hidden_layers',
                                    getattr(self.backbone.config, 'num_layers', 12))
            self.num_heads = getattr(self.backbone.config, 'num_attention_heads',
                                   getattr(self.backbone.config, 'num_heads', 12))
        else:
            self.embed_dim = 768
            self.num_layers = 12
            self.num_heads = 12
        
    def _register_hooks(self):
        def create_hook(layer_idx):
            def hook(module, input, output):
                try:
                    if isinstance(output, tuple) and len(output) > 0:
                        gated = self.head_gate.apply_gate(output[0], layer_idx)
                        return (gated,) + output[1:]
                    else:
                        return self.head_gate.apply_gate(output, layer_idx)
                except:
                    return output
            return hook
        
        try:
            if hasattr(self.backbone, 'encoder'):
                if hasattr(self.backbone.encoder, 'layer'):
                    for i, layer in enumerate(self.backbone.encoder.layer[:self.num_layers]):
                        if hasattr(layer, 'attention') and hasattr(layer.attention, 'self'):
                            layer.attention.self.register_forward_hook(create_hook(i))
                elif hasattr(self.backbone.encoder, 'block'):
                    for i, block in enumerate(self.backbone.encoder.block[:self.num_layers]):
                        if hasattr(block, 'layer') and len(block.layer) > 0:
                            attention_layer = block.layer[0]
                            if hasattr(attention_layer, 'SelfAttention'):
                                attention_layer.SelfAttention.register_forward_hook(create_hook(i))
            elif hasattr(self.backbone, 'transformer') and hasattr(self.backbone.transformer, 'h'):
                for i, layer in enumerate(self.backbone.transformer.h[:self.num_layers]):
                    if hasattr(layer, 'attn'):
                        layer.attn.register_forward_hook(create_hook(i))
        except:
            pass
    
    def get_chunk_embeddings(self, chunks):
        batch_size, num_chunks, chunk_size = chunks.shape
        chunks_flat = chunks.view(-1, chunk_size)
        
        attention_mask = (chunks_flat != self.tokenizer.pad_token_id).long()
        
        with torch.no_grad():
            try:
                if hasattr(self.backbone, 'encoder') and not hasattr(self.backbone, 'decoder'):
                    chunk_outputs = self.backbone.encoder(input_ids=chunks_flat, attention_mask=attention_mask)
                elif hasattr(self.backbone, 'encoder') and hasattr(self.backbone, 'decoder'):
                    chunk_outputs = self.backbone.encoder(input_ids=chunks_flat, attention_mask=attention_mask)
                else:
                    chunk_outputs = self.backbone(input_ids=chunks_flat, attention_mask=attention_mask)
                
                if hasattr(chunk_outputs, 'last_hidden_state'):
                    chunk_embeddings = chunk_outputs.last_hidden_state.mean(dim=1)
                elif hasattr(chunk_outputs, 'hidden_states') and chunk_outputs.hidden_states:
                    chunk_embeddings = chunk_outputs.hidden_states[-1].mean(dim=1)
                else:
                    chunk_embeddings = chunk_outputs[0].mean(dim=1) if isinstance(chunk_outputs, tuple) else chunk_outputs.mean(dim=1)
            except:
                chunk_embeddings = torch.randn(chunks_flat.size(0), self.embed_dim, device=chunks.device)
            
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks, -1)
        return torch.mean(chunk_embeddings, dim=1)
    
    def _get_input_embeddings(self, input_ids):
        try:
            if hasattr(self.backbone, 'embeddings') and hasattr(self.backbone.embeddings, 'word_embeddings'):
                return self.backbone.embeddings.word_embeddings(input_ids)
            elif hasattr(self.backbone, 'shared'):
                return self.backbone.shared(input_ids)
            elif hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'embed_tokens'):
                return self.backbone.encoder.embed_tokens(input_ids)
            else:
                return self.backbone.get_input_embeddings()(input_ids)
        except:
            embedding_layer = nn.Embedding(self.tokenizer.vocab_size, self.embed_dim).to(input_ids.device)
            return embedding_layer(input_ids)
    
    def forward(self, chunks, full_code, attention_mask):
        chunk_embeddings = self.get_chunk_embeddings(chunks)
        prompt_weights = self.router(chunk_embeddings)
        composite_prompt = self.prompt_pool(prompt_weights)
        
        full_code_attention = attention_mask
        prompt_length = composite_prompt.size(1)
        
        if full_code.size(1) + prompt_length > self.config.max_length:
            truncate_length = self.config.max_length - prompt_length
            full_code = full_code[:, :truncate_length]
            full_code_attention = full_code_attention[:, :truncate_length]
        
        inputs_embeds = self._get_input_embeddings(full_code)
        combined_embeds = torch.cat([composite_prompt, inputs_embeds], dim=1)
        
        extended_attention_mask = torch.ones(attention_mask.size(0), prompt_length, 
                                           device=attention_mask.device, dtype=attention_mask.dtype)
        combined_attention_mask = torch.cat([extended_attention_mask, full_code_attention], dim=1)
        
        try:
            if hasattr(self.backbone, 'encoder') and not hasattr(self.backbone, 'decoder'):
                outputs = self.backbone.encoder(inputs_embeds=combined_embeds, attention_mask=combined_attention_mask)
            elif hasattr(self.backbone, 'encoder') and hasattr(self.backbone, 'decoder'):
                outputs = self.backbone.encoder(inputs_embeds=combined_embeds, attention_mask=combined_attention_mask)
            else:
                outputs = self.backbone(inputs_embeds=combined_embeds, attention_mask=combined_attention_mask)
        except:
            outputs = self.backbone(inputs_embeds=combined_embeds, attention_mask=combined_attention_mask, decoder_input_ids=torch.zeros_like(full_code[:, :1]), decoder_attention_mask=torch.ones_like(full_code[:, :1]))
        
        if hasattr(outputs, 'last_hidden_state'):
            sequence_output = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            sequence_output = outputs.hidden_states[-1]
        else:
            sequence_output = outputs[0] if isinstance(outputs, tuple) else outputs
            
        cls_embedding = sequence_output[:, 0, :]
        logits = self.classifier(cls_embedding)
        
        return logits
    
    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
