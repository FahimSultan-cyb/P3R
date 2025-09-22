import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from .components import CompactPromptPool, CompactRouterMLP, CompactHeadGate

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

class UniversalP3RModel(nn.Module):
    def __init__(self, model_name="microsoft/unixcoder-base", num_prompts=4, prompt_length=8, stage=1):
        super().__init__()
        self.model_name = model_name
        self.stage = stage
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.embed_dim = self.backbone.config.hidden_size
        
        if hasattr(self.backbone.config, 'num_hidden_layers'):
            self.num_layers = self.backbone.config.num_hidden_layers
        elif hasattr(self.backbone.config, 'n_layer'):
            self.num_layers = self.backbone.config.n_layer
        else:
            self.num_layers = 12
            
        if hasattr(self.backbone.config, 'num_attention_heads'):
            self.num_heads = self.backbone.config.num_attention_heads
        elif hasattr(self.backbone.config, 'n_head'):
            self.num_heads = self.backbone.config.n_head
        else:
            self.num_heads = 12
        
        self.symbolic_classifier = CompactSymbolicClassifier(self.embed_dim, 2)
        
        self.prompt_pool = CompactPromptPool(num_prompts, prompt_length, self.embed_dim)
        self.router = CompactRouterMLP(self.embed_dim, num_prompts)
        self.head_gate = CompactHeadGate(self.num_layers, self.num_heads)
        
        self.apply_head_gate = False
        self.hook_handles = []
        
        self._set_stage_parameters()
    
    def _set_stage_parameters(self):
        if self.stage == 1:
            for param in self.symbolic_classifier.parameters():
                param.requires_grad = True
            for param in self.prompt_pool.parameters():
                param.requires_grad = False
            for param in self.router.parameters():
                param.requires_grad = False
            for param in self.head_gate.parameters():
                param.requires_grad = False
        elif self.stage == 2:
            for param in self.symbolic_classifier.parameters():
                param.requires_grad = False
            for param in self.prompt_pool.parameters():
                param.requires_grad = True
            for param in self.router.parameters():
                param.requires_grad = True
            for param in self.head_gate.parameters():
                param.requires_grad = True
            self.apply_head_gate = True
            self._register_hooks()
    
    def set_stage(self, stage):
        self.stage = stage
        if hasattr(self, 'hook_handles'):
            for handle in self.hook_handles:
                handle.remove()
            self.hook_handles = []
        self._set_stage_parameters()
    
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
            
        if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
            for i, layer in enumerate(self.backbone.encoder.layer):
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'self'):
                    handle = layer.attention.self.register_forward_hook(create_hook(i))
                    self.hook_handles.append(handle)
        elif hasattr(self.backbone, 'h'):
            for i, layer in enumerate(self.backbone.h):
                if hasattr(layer, 'attn'):
                    handle = layer.attn.register_forward_hook(create_hook(i))
                    self.hook_handles.append(handle)
    
    def get_backbone_embeddings(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states'):
            embeddings = outputs.hidden_states[-1]
        else:
            embeddings = outputs[0]
        return embeddings.mean(dim=1)
    
    def get_chunk_embeddings(self, chunks):
        batch_size, num_chunks, chunk_size = chunks.shape
        chunks_flat = chunks.view(-1, chunk_size)
        attention_mask = (chunks_flat != self.tokenizer.pad_token_id).long()
        
        old_gate_state = self.apply_head_gate
        self.apply_head_gate = False
        with torch.no_grad():
            chunk_embeddings = self.get_backbone_embeddings(chunks_flat, attention_mask)
        self.apply_head_gate = old_gate_state
        
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks, -1)
        return torch.mean(chunk_embeddings, dim=1)
    
    def forward_stage1(self, input_ids, attention_mask):
        embeddings = self.get_backbone_embeddings(input_ids, attention_mask)
        logits = self.symbolic_classifier(embeddings)
        return logits
    
    def forward_stage2(self, chunks, full_code, attention_mask):
        chunk_embeddings = self.get_chunk_embeddings(chunks)
        prompt_weights = self.router(chunk_embeddings)
        composite_prompt = self.prompt_pool(prompt_weights)
        
        full_code_attention = attention_mask
        prompt_length = composite_prompt.size(1)
        
        if full_code.size(1) + prompt_length > 512:
            truncate_length = 512 - prompt_length
            full_code = full_code[:, :truncate_length]
            full_code_attention = full_code_attention[:, :truncate_length]
        
        if hasattr(self.backbone, 'embeddings'):
            inputs_embeds = self.backbone.embeddings(full_code)
        elif hasattr(self.backbone, 'wte'):
            inputs_embeds = self.backbone.wte(full_code)
        else:
            inputs_embeds = self.backbone.get_input_embeddings()(full_code)
        
        combined_embeds = torch.cat([composite_prompt, inputs_embeds], dim=1)
        
        extended_attention_mask = torch.ones(attention_mask.size(0), prompt_length, 
                                           device=attention_mask.device, dtype=attention_mask.dtype)
        combined_attention_mask = torch.cat([extended_attention_mask, full_code_attention], dim=1)
        
        outputs = self.backbone(inputs_embeds=combined_embeds, attention_mask=combined_attention_mask)
        if hasattr(outputs, 'last_hidden_state'):
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        else:
            cls_embedding = outputs[0][:, 0, :]
        
        logits = self.symbolic_classifier(cls_embedding)
        return logits
    
    def forward(self, *args, **kwargs):
        if self.stage == 1:
            return self.forward_stage1(*args, **kwargs)
        else:
            return self.forward_stage2(*args, **kwargs)
    
    def load_stage1_classifier(self, classifier_path):
        classifier_state = torch.load(classifier_path, map_location='cpu')
        self.symbolic_classifier.load_state_dict(classifier_state, strict=False)
        for param in self.symbolic_classifier.parameters():
            param.requires_grad = False
    
    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
