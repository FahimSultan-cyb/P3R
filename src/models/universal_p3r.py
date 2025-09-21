import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from .components import CompactPromptPool, CompactRouterMLP, CompactHeadGate

class UniversalP3RModel(nn.Module):
    def __init__(self, model_name, num_prompts=4, prompt_length=8):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        try:
            self.backbone = AutoModel.from_pretrained(model_name)
        except:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config)
        
        if self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.backbone.resize_token_embeddings(len(self.tokenizer))
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.embed_dim = self.backbone.config.hidden_size
        
        if hasattr(self.backbone.config, 'num_hidden_layers'):
            self.num_layers = self.backbone.config.num_hidden_layers
        elif hasattr(self.backbone.config, 'n_layer'):
            self.num_layers = self.backbone.config.n_layer
        elif hasattr(self.backbone.config, 'num_layers'):
            self.num_layers = self.backbone.config.num_layers
        else:
            self.num_layers = 12
        
        if hasattr(self.backbone.config, 'num_attention_heads'):
            self.num_heads = self.backbone.config.num_attention_heads
        elif hasattr(self.backbone.config, 'n_head'):
            self.num_heads = self.backbone.config.n_head
        else:
            self.num_heads = 12
        
        self.prompt_pool = CompactPromptPool(num_prompts, prompt_length, self.embed_dim)
        self.router = CompactRouterMLP(self.embed_dim, num_prompts)
        self.head_gate = CompactHeadGate(self.num_layers, self.num_heads)
        
        self.classifier = None
        
        self.apply_head_gate = True
        self._register_hooks()
        
    def _register_hooks(self):
        def create_hook(layer_idx):
            def hook(module, input, output):
                if not self.apply_head_gate:
                    return output
                if isinstance(output, tuple) and len(output) > 0:
                    gated = self.head_gate.apply_gate(output[0], layer_idx)
                    return (gated,) + output[1:]
                else:
                    return self.head_gate.apply_gate(output, layer_idx)
            return hook
        
        self.hook_handles = []
        
        if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
            layers = self.backbone.encoder.layer
        elif hasattr(self.backbone, 'transformer') and hasattr(self.backbone.transformer, 'h'):
            layers = self.backbone.transformer.h
        elif hasattr(self.backbone, 'roberta') and hasattr(self.backbone.roberta.encoder, 'layer'):
            layers = self.backbone.roberta.encoder.layer
        elif hasattr(self.backbone, 'bert') and hasattr(self.backbone.bert.encoder, 'layer'):
            layers = self.backbone.bert.encoder.layer
        else:
            return
        
        for i, layer in enumerate(layers):
            if i >= self.num_layers:
                break
            
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'self'):
                handle = layer.attention.self.register_forward_hook(create_hook(i))
            elif hasattr(layer, 'attn'):
                handle = layer.attn.register_forward_hook(create_hook(i))
            else:
                continue
            
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
        
        return chunk_embeddings.view(batch_size, num_chunks, -1).mean(dim=1)
    
    def forward(self, chunks, full_code, attention_mask):
        chunk_embeddings = self.get_chunk_embeddings(chunks)
        prompt_weights = self.router(chunk_embeddings)
        composite_prompt = self.prompt_pool(prompt_weights)
        
        prompt_length = composite_prompt.size(1)
        max_seq_len = 512
        
        if full_code.size(1) + prompt_length > max_seq_len:
            truncate_length = max_seq_len - prompt_length
            full_code = full_code[:, :truncate_length]
            attention_mask = attention_mask[:, :truncate_length]
        
        if hasattr(self.backbone, 'embeddings'):
            inputs_embeds = self.backbone.embeddings(full_code)
        elif hasattr(self.backbone, 'roberta') and hasattr(self.backbone.roberta, 'embeddings'):
            inputs_embeds = self.backbone.roberta.embeddings(full_code)
        elif hasattr(self.backbone, 'bert') and hasattr(self.backbone.bert, 'embeddings'):
            inputs_embeds = self.backbone.bert.embeddings(full_code)
        elif hasattr(self.backbone, 'transformer') and hasattr(self.backbone.transformer, 'wte'):
            inputs_embeds = self.backbone.transformer.wte(full_code)
        else:
            inputs_embeds = self.backbone.get_input_embeddings()(full_code)
        
        combined_embeds = torch.cat([composite_prompt, inputs_embeds], dim=1)
        
        extended_attention_mask = torch.ones(attention_mask.size(0), prompt_length, 
                                           device=attention_mask.device, dtype=attention_mask.dtype)
        combined_attention_mask = torch.cat([extended_attention_mask, attention_mask], dim=1)
        
        outputs = self.backbone(inputs_embeds=combined_embeds, attention_mask=combined_attention_mask)
        
        if hasattr(outputs, 'last_hidden_state'):
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, 'hidden_states'):
            cls_embedding = outputs.hidden_states[-1][:, 0, :]
        else:
            cls_embedding = outputs[0][:, 0, :]
        
        return cls_embedding
    
    def predict(self, chunks, full_code, attention_mask):
        if self.classifier is None:
            raise RuntimeError("No classifier attached. Use after Stage 1 training.")
        
        embeddings = self.forward(chunks, full_code, attention_mask)
        logits = self.classifier(embeddings)
        return logits
    
    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total