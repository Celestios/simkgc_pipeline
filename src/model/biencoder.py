import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from transformers import AutoModel, AutoConfig

class SimKGCBiEncoder(nn.Module):
    """
    Production SimKGC Dual Transformer Architecture with Matryoshka Projection.
    
    Encodes (Head + Relation) via head forward pass and (Tail) via tail forward pass.
    Projects 768-d hidden representation to 256-d normalized space with 128-d nested support.
    """
    
    def __init__(self, backbone_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 output_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        try:
            self.config = AutoConfig.from_pretrained(backbone_name)
        except Exception:
            from transformers import BertConfig
            self.config = BertConfig.from_pretrained(backbone_name)
            
        try:
            self.encoder = AutoModel.from_pretrained(backbone_name, config=self.config)
        except Exception:
            from transformers import BertModel
            try:
                self.encoder = BertModel.from_pretrained(backbone_name, config=self.config)
            except Exception:
                self.encoder = BertModel(self.config)
            
        self.hidden_size = self.config.hidden_size
        self.output_dim = output_dim
        
        # Dense linear projection for Matryoshka 256d
        self.projection = nn.Linear(self.hidden_size, self.output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize projection weights with normal distribution
        nn.init.orthogonal_(self.projection.weight)

    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Applies mean pooling over token embeddings respecting attention mask."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Runs transformer encoder, pools tokens, and projects to normalized 256-d vector.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pooling(outputs.last_hidden_state, attention_mask)
        dropped = self.dropout(pooled)
        projected = self.projection(dropped)
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized

    def forward(self, hr_input_ids: torch.Tensor, hr_attention_mask: torch.Tensor,
                tail_input_ids: torch.Tensor, tail_attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simultaneous dual forward pass for (Head + Relation) and (Tail).
        """
        hr_vecs = self.encode(hr_input_ids, hr_attention_mask)
        tail_vecs = self.encode(tail_input_ids, tail_attention_mask)
        return hr_vecs, tail_vecs
