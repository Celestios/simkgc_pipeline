#!/usr/bin/env python3
"""
Modular Transformer Architecture for SimKGC Knowledge Graph Completion.

Decoupled modular design optimized for Kaggle GPU environments:
  1. TextEmbedder (Layers 1–8): Direct surface text -> 256d BGE-M3 state space alignment.
  2. RelationalCore (Layers 9–12): Direct manifold reasoning on 256d vectors (v_h* + r -> v_t*).
  3. AssembledBiEncoder: Unified end-to-end model connecting Layer 8 to Layer 9 via 384d highway.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Union
from transformers import AutoModel, AutoConfig, BertConfig, BertModel

def mean_pooling(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Standard attention-masked mean pooling across token sequence."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    sum_embeddings = torch.sum(hidden_state * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def format_4d_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Creates (BatchSize, 1, SeqLen, SeqLen) additive mask for universal SDPA/Eager compatibility."""
    seq_len = attention_mask.size(1)
    expanded = attention_mask[:, None, None, :].expand(-1, 1, seq_len, -1).to(dtype=dtype)
    return (1.0 - expanded) * -10000.0

class TextEmbedder(nn.Module):
    """
    Sub-network for Layers 1–8 (Text-to-Space Embedder).
    Maps arbitrary Persian/English concept strings directly into 256-d BGE-M3 coordinate space.
    """
    def __init__(self,
                 backbone_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 output_dim: int = 256,
                 split_layer: int = 8,
                 dropout: float = 0.1,
                 backbone_model: Optional[nn.Module] = None):
        super().__init__()
        self.backbone_name = backbone_name
        self.output_dim = output_dim
        self.split_layer = split_layer

        if backbone_model is not None:
            full_model = backbone_model
            self.config = getattr(full_model, "config", None)
        else:
            try:
                self.config = AutoConfig.from_pretrained(backbone_name)
            except Exception:
                self.config = BertConfig.from_pretrained(backbone_name)

            try:
                full_model = AutoModel.from_pretrained(backbone_name, config=self.config)
            except Exception:
                full_model = BertModel.from_pretrained(backbone_name, config=self.config)

        self.vocab_size = getattr(self.config, "vocab_size", 250002)
        self.hidden_size = getattr(self.config, "hidden_size", 384)

        # Embeddings & Layers 0..split_layer-1
        self.embeddings = full_model.embeddings
        self.layers = nn.ModuleList([full_model.encoder.layer[i] for i in range(split_layer)])
        
        # Projection head: 384d -> 256d
        self.projection = nn.Linear(self.hidden_size, self.output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.orthogonal_(self.projection.weight)

    def get_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes input through embeddings and Layers 1–8.
        Returns:
            hidden_states: (BatchSize, SeqLen, 384)
            extended_attention_mask: (BatchSize, 1, SeqLen, SeqLen)
        """
        safe_input_ids = torch.clamp(input_ids, min=0, max=self.vocab_size - 1)
        hidden_states = self.embeddings(safe_input_ids)
        extended_attention_mask = format_4d_attention_mask(attention_mask, hidden_states.dtype)

        for layer in self.layers:
            layer_outputs = layer(hidden_states, attention_mask=extended_attention_mask)
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        return hidden_states, extended_attention_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encodes text to normalized 256-d vector space."""
        hidden_states, _ = self.get_hidden_states(input_ids, attention_mask)
        pooled = mean_pooling(hidden_states, attention_mask)
        dropped = self.dropout(pooled)
        projected = self.projection(dropped)
        return F.normalize(projected, p=2, dim=-1)


class RelationalCore(nn.Module):
    """
    Sub-network for Layers 9–12 (Relational Reasoning Core).
    Performs pure graph manifold transitions (v_h* + r -> v_t*) on 256-d vectors.
    """
    def __init__(self,
                 backbone_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 input_dim: int = 256,
                 output_dim: int = 256,
                 num_relations: int = 32,
                 split_layer: int = 8,
                 total_layers: int = 12,
                 dropout: float = 0.1,
                 backbone_model: Optional[nn.Module] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.split_layer = split_layer
        self.total_layers = total_layers

        if backbone_model is not None:
            full_model = backbone_model
            self.config = getattr(full_model, "config", None)
        else:
            try:
                self.config = AutoConfig.from_pretrained(backbone_name)
            except Exception:
                self.config = BertConfig.from_pretrained(backbone_name)

            try:
                full_model = AutoModel.from_pretrained(backbone_name, config=self.config)
            except Exception:
                full_model = BertModel.from_pretrained(backbone_name, config=self.config)

        self.hidden_size = getattr(self.config, "hidden_size", 384)

        # Input adapter for 256d teacher vectors -> 384d transformer space
        self.input_proj = nn.Linear(self.input_dim, self.hidden_size, bias=False)
        nn.init.orthogonal_(self.input_proj.weight)

        # Relation embeddings in 384d transformer space
        self.relation_embeddings = nn.Embedding(num_relations, self.hidden_size)
        nn.init.normal_(self.relation_embeddings.weight, mean=0.0, std=0.02)

        # Extract Layers 9–12 (split_layer..total_layers-1)
        self.layers = nn.ModuleList([full_model.encoder.layer[i] for i in range(split_layer, total_layers)])

        # Output projector: 384d -> 256d
        self.output_proj = nn.Linear(self.hidden_size, self.output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.orthogonal_(self.output_proj.weight)

    def forward(self, head_vecs: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        """
        Primary PyTorch Module Interface (Vector Mode).
        Takes 256d head vectors and relation IDs, predicts 256d tail vectors.
        """
        batch_size = head_vecs.size(0)

        # Project head vector: 256d -> 384d (Shape: B, 1, 384)
        h_proj = self.input_proj(head_vecs).unsqueeze(1)

        # Lookup relation embedding (Shape: B, 1, 384)
        r_embed = self.relation_embeddings(relation_ids).unsqueeze(1)

        # Concatenate sequence: [Head, Relation] (Shape: B, 2, 384)
        hidden_states = torch.cat([h_proj, r_embed], dim=1)

        # Full self-attention across [Head, Relation] tokens: 2 valid tokens
        attention_mask = torch.ones((batch_size, 2), device=head_vecs.device, dtype=torch.long)
        extended_attention_mask = format_4d_attention_mask(attention_mask, hidden_states.dtype)

        for layer in self.layers:
            layer_outputs = layer(hidden_states, attention_mask=extended_attention_mask)
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        # Mean pool representation of the pair
        pooled = hidden_states.mean(dim=1)
        dropped = self.dropout(pooled)
        projected = self.output_proj(dropped)
        return F.normalize(projected, p=2, dim=-1)

    # Alias for explicit clarity
    forward_vectors = forward

    def forward_hidden_states(self, hidden_states: torch.Tensor,
                              attention_mask: torch.Tensor,
                              extended_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Highway Mode: Takes 384d hidden states from Layer 8, processes through Layers 9–12.
        """
        for layer in self.layers:
            layer_outputs = layer(hidden_states, attention_mask=extended_attention_mask)
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        pooled = mean_pooling(hidden_states, attention_mask)
        dropped = self.dropout(pooled)
        projected = self.output_proj(dropped)
        return F.normalize(projected, p=2, dim=-1)


class AssembledBiEncoder(nn.Module):
    """
    Unified Production Dual Encoder assembling TextEmbedder (L1–8) and RelationalCore (L9–12).
    Connects Layer 8 directly to Layer 9 over the 384-d highway without dimensional bottleneck.
    Implements standard encode() interface matching SimKGCBiEncoder for seamless polymorphism.
    """
    def __init__(self,
                 text_embedder: TextEmbedder,
                 relational_core: RelationalCore):
        super().__init__()
        self.text_embedder = text_embedder
        self.relational_core = relational_core
        self.vocab_size = text_embedder.vocab_size
        self.output_dim = relational_core.output_dim

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard polymorphic text encoding pass: Text -> L1-8 -> L9-12 -> 256d."""
        hidden_states, extended_attention_mask = self.text_embedder.get_hidden_states(input_ids, attention_mask)
        return self.relational_core.forward_hidden_states(hidden_states, attention_mask, extended_attention_mask)

    # Alias for backward compatibility
    encode_text = encode

    def encode_vector(self, head_vecs: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        """Vector Bypass Mode: Pre-computed Concept Vector -> Input Adapter -> Layers 9–12 -> Output (256d)."""
        return self.relational_core(head_vecs, relation_ids)

    def forward(self,
                hr_input_ids: torch.Tensor,
                hr_attention_mask: torch.Tensor,
                tail_input_ids: torch.Tensor,
                tail_attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dual forward pass for (Head + Relation) query and (Tail) target."""
        hr_vecs = self.encode(hr_input_ids, hr_attention_mask)
        tail_vecs = self.encode(tail_input_ids, tail_attention_mask)
        return hr_vecs, tail_vecs
