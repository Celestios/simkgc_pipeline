import unittest
import torch
from transformers import BertConfig, BertModel
from src.model.biencoder import SimKGCBiEncoder

class TestSimKGCModel(unittest.TestCase):
    def setUp(self):
        self.config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=64
        )
        self.base_model = BertModel(self.config)
        # Proper initialization with backbone injection (DIP)
        self.model = SimKGCBiEncoder(output_dim=256, backbone_model=self.base_model)

    def test_forward_output_shapes_and_normalization(self):
        batch_size = 4
        seq_len = 16
        hr_ids = torch.randint(0, 1000, (batch_size, seq_len))
        hr_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        t_ids = torch.randint(0, 1000, (batch_size, seq_len))
        t_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        hr_vecs, tail_vecs = self.model(hr_ids, hr_mask, t_ids, t_mask)
        
        # Test shape
        self.assertEqual(list(hr_vecs.shape), [batch_size, 256])
        self.assertEqual(list(tail_vecs.shape), [batch_size, 256])

        # Test L2 normalization
        hr_norms = torch.norm(hr_vecs, dim=-1)
        tail_norms = torch.norm(tail_vecs, dim=-1)
        self.assertTrue(torch.allclose(hr_norms, torch.ones(batch_size), atol=1e-3))
        self.assertTrue(torch.allclose(tail_norms, torch.ones(batch_size), atol=1e-3))

if __name__ == "__main__":
    unittest.main()
