import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from .components import CompactPromptPool, CompactRouterMLP, CompactHeadGate

class P3RStage2Model(nn.Module):
    def __init__(self, model_name, stage1_classifier_path, num_prompts=4, prompt_length=8):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.embed_dim = self.backbone.config.hidden_size
        self.num_layers = self.backbone.config.num_hidden_layers
        self.num_heads = self.backbone.config.num_attention_heads
        
        self.prompt_pool = CompactPromptPool(num_prompts, prompt_length, self.embed_dim)
        self.router = CompactRouterMLP(self.embed_dim, num_prompts)
        self.head_gate = CompactHeadGate(self.num_layers, self.num_heads)
        
        from ..training.stage1_trainer import CompactSymbolicClassifier
        self.frozen_classifier = CompactSymbolicClassifier(self.embed_dim, 2)
        self.frozen_classifier.load_state_dict(torch.load(stage1_classifier_path))
        
        for param in self.frozen_classifier.parameters():
            param.requires_grad = False
        
        self.apply_head_gate = True
        self._register_hooks()
        
    def _register_hooks(self):
        def create_hook(layer_idx):
            def hook(module, input, output):
                if not self.apply_head_gate:
                    return output
                if isinstance(output, tuple) and len(output) > 0:
                    gated = self.head_gate.apply_gate(output[0], layer_idx)
                    output = (gated,) + output[1:]
                    return output
                else:
                    return self.head_gate.apply_gate(output, layer_idx)
            return hook
            
        self.hook_handles = []
        for i, layer in enumerate(self.backbone.encoder.layer):
            handle = layer.attention.self.register_forward_hook(create_hook(i))
            self.hook_handles.append(handle)
    
    def get_chunk_embeddings(self, chunks):
        batch_size, num_chunks, chunk_size = chunks.shape
        chunks_flat = chunks.view(-1, chunk_size)
        attention_mask = (chunks_flat != self.tokenizer.pad_token_id).long()
        
        self.apply_head_gate = False
        with torch.no_grad():
            chunk_outputs = self.backbone(input_ids=chunks_flat, attention_mask=attention_mask)
            chunk_embeddings = chunk_outputs.last_hidden_state.mean(dim=1)
        self.apply_head_gate = True
        
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
        
        inputs_embeds = self.backbone.embeddings(full_code)
        combined_embeds = torch.cat([composite_prompt, inputs_embeds], dim=1)
        
        extended_attention_mask = torch.ones(attention_mask.size(0), prompt_length, 
                                           device=attention_mask.device, dtype=attention_mask.dtype)
        combined_attention_mask = torch.cat([extended_attention_mask, full_code_attention], dim=1)
        
        outputs = self.backbone(inputs_embeds=combined_embeds, attention_mask=combined_attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        logits = self.frozen_classifier(cls_embedding)
        return logits
    
    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
