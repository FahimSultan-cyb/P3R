import torch
import torch.nn as nn

class CompactSymbolicClassifier(nn.Module):
    def __init__(self, embed_dim=768, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

class CompactPromptPool(nn.Module):
    def __init__(self, num_prompts=4, prompt_length=8, embed_dim=768):
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

class CompactRouterMLP(nn.Module):
    def __init__(self, embed_dim=768, num_prompts=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 384),
            nn.ReLU(),
            nn.Linear(384, num_prompts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, embeddings):
        return self.mlp(embeddings)

class CompactHeadGate(nn.Module):
    def __init__(self, num_layers=12, num_heads=12):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.gates = nn.Parameter(torch.ones(num_layers, num_heads) * 0.8)
        
    def apply_gate(self, attention_output, layer_idx):
        if attention_output.dim() == 3:
            batch_size, seq_len, hidden_size = attention_output.shape
            head_size = hidden_size // self.num_heads
            attention_output = attention_output.view(batch_size, seq_len, self.num_heads, head_size)
            gated_output = attention_output * self.gates[layer_idx].view(1, 1, self.num_heads, 1)
            return gated_output.view(batch_size, seq_len, hidden_size)
        return attention_output