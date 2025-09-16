import unittest
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.p3r_model import P3RHeadGateModel
from src.models.components import CompactPromptPool, CompactRouterMLP, CompactHeadGate

class TestP3RModel(unittest.TestCase):
    
    def setUp(self):
        self.model = P3RHeadGateModel()
        self.batch_size = 2
        self.seq_length = 128
        self.num_chunks = 3
        
    def test_model_initialization(self):
        self.assertIsNotNone(self.model.tokenizer)
        self.assertIsNotNone(self.model.backbone)
        self.assertIsNotNone(self.model.prompt_pool)
        self.assertIsNotNone(self.model.router)
        
    def test_parameter_efficiency(self):
        trainable, total = self.model.count_parameters()
        efficiency_ratio = trainable / total
        self.assertLess(efficiency_ratio, 0.05)  # Less than 5% trainable
        
    def test_forward_pass(self):
        chunks = torch.randint(0, 1000, (self.batch_size, self.num_chunks, self.seq_length))
        full_code = torch.randint(0, 1000, (self.batch_size, self.seq_length))
        attention_mask = torch.ones(self.batch_size, self.seq_length)
        
        with torch.no_grad():
            output = self.model(chunks, full_code, attention_mask)
            
        self.assertEqual(output.shape, (self.batch_size, 2))
        
    def test_component_dimensions(self):
        prompt_pool = CompactPromptPool(num_prompts=4, prompt_length=8, embed_dim=768)
        router = CompactRouterMLP(embed_dim=768, num_prompts=4)
        head_gate = CompactHeadGate(num_layers=12, num_heads=12)
        
        # Test prompt pool
        weights = torch.rand(self.batch_size, 4)
        prompt_output = prompt_pool(weights)
        self.assertEqual(prompt_output.shape, (self.batch_size, 8, 768))
        
        # Test router
        embeddings = torch.rand(self.batch_size, 768)
        router_output = router(embeddings)
        self.assertEqual(router_output.shape, (self.batch_size, 4))
        self.assertTrue(torch.allclose(router_output.sum(dim=1), torch.ones(self.batch_size)))

if __name__ == '__main__':
    unittest.main()