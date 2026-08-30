import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SimKGCMatryoshkaLoss(nn.Module):
    """
    Standard SimKGC InfoNCE Loss (ACL 2022).
    Computes symmetric bidirectional contrastive loss over in-batch negatives in full 256-d space.
    """
    
    def __init__(self, temperature: float = 0.05, primary_dim: int = 256, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.primary_dim = primary_dim
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, hr_embeddings: torch.Tensor, tail_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hr_embeddings: Tensor of shape (BatchSize, 256)
            tail_embeddings: Tensor of shape (BatchSize, 256)
        """
        batch_size = hr_embeddings.size(0)
        labels = torch.arange(batch_size, device=hr_embeddings.device, dtype=torch.long)
        
        # Symmetric In-Batch Contrastive Loss (ACL 2022 formulation)
        sim_matrix = torch.matmul(hr_embeddings, tail_embeddings.T) / self.temperature
        loss_hr_to_t = self.cross_entropy(sim_matrix, labels)
        loss_t_to_hr = self.cross_entropy(sim_matrix.T, labels)
        
        return (loss_hr_to_t + loss_t_to_hr) / 2.0

class SimKGCDistillationLoss(nn.Module):
    """
    Teacher-Student Distillation Loss without artificial sub-vector slicing.
    Focuses 100% of gradient capacity on full 256-d alignment with BGE-M3 teacher targets.
    """
    def __init__(self, temperature: float = 0.05, alpha: float = 0.5, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Balance between direct cosine regression and in-batch contrastive matching
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
        
        total_loss = (self.alpha * loss_align) + ((1.0 - self.alpha) * loss_contrast)
        return total_loss
