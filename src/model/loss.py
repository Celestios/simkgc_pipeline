import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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

class SimKGCDistillationLoss(nn.Module):
    """
    Teacher-Student Distillation Loss.
    Aligns student query representations directly into the teacher's (BGE-M3) coordinate space
    using cosine regression and contrastive target matching.
    """
    def __init__(self, temperature: float = 0.05, alpha: float = 0.7, aux_dim: int = 128):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for direct teacher cosine alignment vs contrastive
        self.aux_dim = aux_dim
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
        
        # 1. Direct Cosine Distance Alignment (Pull Student HR directly to Teacher Target)
        cosine_sim = torch.sum(student_hr_vecs * teacher_tail_vecs, dim=-1)
        loss_align = torch.mean(1.0 - cosine_sim)
        
        # If student tail vecs are provided, align them to teacher targets as well
        if student_tail_vecs is not None:
            tail_cosine_sim = torch.sum(student_tail_vecs * teacher_tail_vecs, dim=-1)
            loss_align = (loss_align + torch.mean(1.0 - tail_cosine_sim)) / 2.0
            
        # 2. Contrastive Target Matching (Discriminate against other teacher targets in batch)
        contrastive_matrix = torch.matmul(student_hr_vecs, teacher_tail_vecs.T) / self.temperature
        loss_contrast = self.cross_entropy(contrastive_matrix, labels)
        
        # 3. Matryoshka 128-d Auxiliary Alignment
        hr_128 = F.normalize(student_hr_vecs[:, :self.aux_dim], p=2, dim=-1)
        teacher_128 = F.normalize(teacher_tail_vecs[:, :self.aux_dim], p=2, dim=-1)
        loss_aux = torch.mean(1.0 - torch.sum(hr_128 * teacher_128, dim=-1))
        
        total_loss = (self.alpha * loss_align) + ((1.0 - self.alpha) * loss_contrast) + (0.3 * loss_aux)
        return total_loss
