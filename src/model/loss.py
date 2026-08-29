import torch
import torch.nn as nn
import torch.nn.functional as F

class SimKGCMatryoshkaLoss(nn.Module):
    """
    SimKGC InfoNCE Loss with Matryoshka Representation Learning (MRL).
    
    Computes symmetric contrastive loss over in-batch negatives for primary dimension (256d)
    and nested auxiliary dimension (128d).
    """
    
    def __init__(self, temperature: float = 0.05,
                 primary_dim: int = 256,
                 aux_dim: int = 128,
                 aux_weight: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.primary_dim = primary_dim
        self.aux_dim = aux_dim
        self.aux_weight = aux_weight
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, hr_embeddings: torch.Tensor, tail_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hr_embeddings: Tensor of shape (BatchSize, 256)
            tail_embeddings: Tensor of shape (BatchSize, 256)
        """
        batch_size = hr_embeddings.size(0)
        labels = torch.arange(batch_size, device=hr_embeddings.device, dtype=torch.long)
        
        # 1. Primary Dimension (256d) Similarity Matrix
        sim_matrix_256 = torch.matmul(hr_embeddings, tail_embeddings.T) / self.temperature
        loss_hr_to_t_256 = self.cross_entropy(sim_matrix_256, labels)
        loss_t_to_hr_256 = self.cross_entropy(sim_matrix_256.T, labels)
        loss_256 = (loss_hr_to_t_256 + loss_t_to_hr_256) / 2.0
        
        # 2. Nested Dimension (128d) Matryoshka Loss
        hr_128 = F.normalize(hr_embeddings[:, :self.aux_dim], p=2, dim=-1)
        tail_128 = F.normalize(tail_embeddings[:, :self.aux_dim], p=2, dim=-1)
        sim_matrix_128 = torch.matmul(hr_128, tail_128.T) / self.temperature
        loss_hr_to_t_128 = self.cross_entropy(sim_matrix_128, labels)
        loss_t_to_hr_128 = self.cross_entropy(sim_matrix_128.T, labels)
        loss_128 = (loss_hr_to_t_128 + loss_t_to_hr_128) / 2.0
        
        # Total combined loss
        total_loss = loss_256 + self.aux_weight * loss_128
        return total_loss
