import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union

class SimKGCMatryoshkaLoss(nn.Module):
    """
    SimKGC InfoNCE Loss with nested Matryoshka Representation Learning (MRL) support.
    Computes symmetric bidirectional contrastive loss over in-batch negatives across
    primary (e.g. 256d) and optional auxiliary nested dimensions (e.g. 128d, 64d).
    """
    def __init__(self,
                 temperature: float = 0.05,
                 primary_dim: int = 256,
                 aux_dim: Optional[Union[int, List[int]]] = None,
                 mrl_weight: float = 0.3,
                 **kwargs):
        super().__init__()
        self.temperature = temperature
        self.primary_dim = primary_dim
        self.mrl_weight = mrl_weight
        if isinstance(aux_dim, int):
            self.aux_dims = [aux_dim]
        elif isinstance(aux_dim, list):
            self.aux_dims = aux_dim
        else:
            self.aux_dims = []
            
        self.cross_entropy = nn.CrossEntropyLoss()

    def _compute_infonce(self, hr: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        batch_size = hr.size(0)
        labels = torch.arange(batch_size, device=hr.device, dtype=torch.long)
        sim_matrix = torch.matmul(hr, tail.T) / self.temperature
        loss_hr_to_t = self.cross_entropy(sim_matrix, labels)
        loss_t_to_hr = self.cross_entropy(sim_matrix.T, labels)
        return (loss_hr_to_t + loss_t_to_hr) / 2.0

    def forward(self, hr_embeddings: torch.Tensor, tail_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hr_embeddings: Tensor of shape (BatchSize, primary_dim)
            tail_embeddings: Tensor of shape (BatchSize, primary_dim)
        """
        # Primary full-dimensional loss
        loss = self._compute_infonce(hr_embeddings, tail_embeddings)

        # MRL nested sub-vector losses
        for dim in self.aux_dims:
            if dim < hr_embeddings.size(-1):
                hr_slice = F.normalize(hr_embeddings[:, :dim], p=2, dim=-1)
                tail_slice = F.normalize(tail_embeddings[:, :dim], p=2, dim=-1)
                loss += self.mrl_weight * self._compute_infonce(hr_slice, tail_slice)

        return loss

class SimKGCDistillationLoss(nn.Module):
    """
    Teacher-Student Distillation Loss.
    Combines direct cosine distance alignment with in-batch contrastive matching
    against frozen BGE-M3 teacher target vectors.
    """
    def __init__(self, temperature: float = 0.05, alpha: float = 0.5, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Balance: alpha * cosine_alignment + (1 - alpha) * contrastive
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, student_hr_vecs: torch.Tensor,
                student_tail_vecs: Optional[torch.Tensor],
                teacher_tail_vecs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_hr_vecs: (BatchSize, 256) predicted query vector from student
            student_tail_vecs: (BatchSize, 256) optional student tail vector
            teacher_tail_vecs: (BatchSize, 256) frozen target vector from BGE-M3 teacher
        """
        batch_size = student_hr_vecs.size(0)
        labels = torch.arange(batch_size, device=student_hr_vecs.device, dtype=torch.long)

        # 1. Direct Cosine Distance Alignment (Pull Student HR directly onto Teacher Target)
        cosine_sim = torch.sum(student_hr_vecs * teacher_tail_vecs, dim=-1)
        loss_align = torch.mean(1.0 - cosine_sim)

        if student_tail_vecs is not None:
            tail_cosine_sim = torch.sum(student_tail_vecs * teacher_tail_vecs, dim=-1)
            loss_align = (loss_align + torch.mean(1.0 - tail_cosine_sim)) / 2.0

        # 2. Contrastive Target Matching (Discriminate against other teacher targets in batch)
        contrastive_matrix = torch.matmul(student_hr_vecs, teacher_tail_vecs.T) / self.temperature
        loss_contrast = self.cross_entropy(contrastive_matrix, labels)

        return (self.alpha * loss_align) + ((1.0 - self.alpha) * loss_contrast)
