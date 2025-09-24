import torch
from dataclasses import dataclass

@dataclass
class P3RConfig:
    model_name: str = "microsoft/unixcoder-base"
    num_prompts: int = 4
    prompt_length: int = 8
    num_classes: int = 2
    max_length: int = 512
    chunk_size: int = 512
    stride: int = 256
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 5
    dropout: float = 0.1
    device: str = "auto"
    code_col: str = "func"
    label_col: str = "label"
    
    def __post_init__(self):
        if self.device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
