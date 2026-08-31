import unittest
import torch
from transformers import BertConfig, BertModel
from src.model.modular_encoder import TextEmbedder, RelationalCore, AssembledBiEncoder

class TestModularEncoder(unittest.TestCase):
    def setUp(self):
        self.config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=2,
            max_position_embeddings=128
        )
        self.base_model = BertModel(self.config)

        # Real constructor initialization with backbone injection (DIP)
        self.embedder = TextEmbedder(
            output_dim=256,
            split_layer=2,
            backbone_model=self.base_model
        )

        self.core = RelationalCore(
            input_dim=256,
            output_dim=256,
            num_relations=32,
            split_layer=2,
            total_layers=4,
            backbone_model=self.base_model
        )

        self.assembled = AssembledBiEncoder(self.embedder, self.core)

    def test_text_embedder_forward_and_norm(self):
        batch_size, seq_len = 4, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        vecs = self.embedder(input_ids, mask)
        self.assertEqual(list(vecs.shape), [batch_size, 256])

        # Verify unit norm
        norms = torch.norm(vecs, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(batch_size), atol=1e-3))

    def test_relational_core_forward_contract_and_norm(self):
        batch_size = 4
        head_vecs = torch.randn(batch_size, 256)
        head_vecs = torch.nn.functional.normalize(head_vecs, p=2, dim=-1)
        relation_ids = torch.randint(0, 32, (batch_size,))

        # Test standard forward() (PyTorch nn.Module contract)
        pred_tail_vecs = self.core(head_vecs, relation_ids)
        self.assertEqual(list(pred_tail_vecs.shape), [batch_size, 256])

        # Verify unit norm
        norms = torch.norm(pred_tail_vecs, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(batch_size), atol=1e-3))

    def test_assembled_biencoder_encode_interface(self):
        batch_size, seq_len = 4, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        # Test polymorphic encode() matching SimKGCBiEncoder
        encoded = self.assembled.encode(input_ids, mask)
        self.assertEqual(list(encoded.shape), [batch_size, 256])
        self.assertTrue(torch.allclose(torch.norm(encoded, dim=-1), torch.ones(batch_size), atol=1e-3))

    def test_assembled_biencoder_dual_text_pass(self):
        batch_size, seq_len = 4, 16
        hr_ids = torch.randint(0, 1000, (batch_size, seq_len))
        hr_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        t_ids = torch.randint(0, 1000, (batch_size, seq_len))
        t_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        hr_vecs, tail_vecs = self.assembled(hr_ids, hr_mask, t_ids, t_mask)
        self.assertEqual(list(hr_vecs.shape), [batch_size, 256])
        self.assertEqual(list(tail_vecs.shape), [batch_size, 256])

        self.assertTrue(torch.allclose(torch.norm(hr_vecs, dim=-1), torch.ones(batch_size), atol=1e-3))
        self.assertTrue(torch.allclose(torch.norm(tail_vecs, dim=-1), torch.ones(batch_size), atol=1e-3))

    def test_assembled_vector_bypass(self):
        batch_size = 4
        head_vecs = torch.randn(batch_size, 256)
        head_vecs = torch.nn.functional.normalize(head_vecs, p=2, dim=-1)
        relation_ids = torch.randint(0, 32, (batch_size,))

        vec_out = self.assembled.encode_vector(head_vecs, relation_ids)
        self.assertEqual(list(vec_out.shape), [batch_size, 256])
        self.assertTrue(torch.allclose(torch.norm(vec_out, dim=-1), torch.ones(batch_size), atol=1e-3))

if __name__ == "__main__":
    unittest.main()
