import unittest
import torch
from src.model.loss import SimKGCMatryoshkaLoss

class TestSimKGCLoss(unittest.TestCase):
    def setUp(self):
        self.criterion = SimKGCMatryoshkaLoss(temperature=0.05, primary_dim=256, aux_dim=128)

    def test_loss_computation_and_backward(self):
        batch_size = 8
        # Create random normalized 256d vectors
        hr_vecs = torch.randn(batch_size, 256, requires_grad=True)
        hr_norm = torch.nn.functional.normalize(hr_vecs, p=2, dim=-1)

        tail_vecs = torch.randn(batch_size, 256, requires_grad=True)
        tail_norm = torch.nn.functional.normalize(tail_vecs, p=2, dim=-1)

        loss = self.criterion(hr_norm, tail_norm)
        
        # Loss must be a positive scalar
        self.assertGreater(loss.item(), 0.0)
        
        # Backward pass must populate gradients
    def test_distillation_loss(self):
        from src.model.loss import SimKGCDistillationLoss
        distill_criterion = SimKGCDistillationLoss(temperature=0.05, alpha=0.7)
        batch_size = 8
        
        hr_vecs = torch.nn.functional.normalize(torch.randn(batch_size, 256, requires_grad=True), p=2, dim=-1)
        tail_vecs = torch.nn.functional.normalize(torch.randn(batch_size, 256, requires_grad=True), p=2, dim=-1)
        teacher_targets = torch.nn.functional.normalize(torch.randn(batch_size, 256), p=2, dim=-1)
        
        loss = distill_criterion(hr_vecs, tail_vecs, teacher_targets)
        self.assertGreater(loss.item(), 0.0)
        
        loss.backward()
        self.assertIsNotNone(hr_vecs.grad)
        self.assertIsNotNone(tail_vecs.grad)

if __name__ == "__main__":
    unittest.main()
