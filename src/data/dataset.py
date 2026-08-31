import sys
import json
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.data.relations import format_verbalizer_prompt
except ImportError:
    from relations import format_verbalizer_prompt

class SimKGCDataset(Dataset):
    """
    PyTorch Dataset for Knowledge Graph Completion.
    Loads relational triplets (head, relation, tail) from JSON files.
    """
    def __init__(self, data_paths: List[str]):
        self.triples = []
        for path in data_paths:
            p = Path(path)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    items = json.load(f)
                    for item in items:
                        if "head" in item and "relation" in item and "tail" in item:
                            self.triples.append((item["head"], item["relation"], item["tail"]))
        if len(self.triples) == 0:
            raise ValueError(f"No valid triplets found in provided paths: {data_paths}")
            
    def __len__(self) -> int:
        return len(self.triples)
        
    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        return self.triples[idx]


class SimKGCCollator:
    """
    Collates raw string triplets into tokenized PyTorch tensors.
    Uses fluent natural language verbalizer prompts.
    Optionally attaches teacher concept target indices for zero-overhead distillation.
    """
    def __init__(self, tokenizer, max_seq_length: int = 64, max_length: Optional[int] = None,
                 concept_to_idx: Optional[Dict[str, int]] = None, **kwargs):
        self.tokenizer = tokenizer
        self.max_seq_length = max_length if max_length is not None else max_seq_length
        self.concept_to_idx = concept_to_idx

    def __call__(self, batch: List[Tuple[str, str, str]]) -> Dict[str, torch.Tensor]:
        hr_texts = [format_verbalizer_prompt(head, relation) for head, relation, _ in batch]
        tail_texts = [tail for _, _, tail in batch]
        
        hr_encodings = self.tokenizer(
            hr_texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        tail_encodings = self.tokenizer(
            tail_texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        batch_dict = {
            "hr_input_ids": hr_encodings["input_ids"],
            "hr_attention_mask": hr_encodings["attention_mask"],
            "tail_input_ids": tail_encodings["input_ids"],
            "tail_attention_mask": tail_encodings["attention_mask"]
        }
        
        if self.concept_to_idx is not None:
            tail_indices = [self.concept_to_idx.get(tail, -1) for tail in tail_texts]
            batch_dict["tail_indices"] = torch.tensor(tail_indices, dtype=torch.long)
            
        return batch_dict


class ConceptDataset(Dataset):
    """Dataset of unique concepts and their pre-computed teacher embeddings (Stage 1A)."""
    def __init__(self, concepts: List[str], teacher_embeddings: torch.Tensor, tokenizer=None, max_length: int = 32):
        self.concepts = concepts
        self.teacher_embeddings = teacher_embeddings
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.concepts)

    def __getitem__(self, idx):
        concept = self.concepts[idx]
        target = self.teacher_embeddings[idx]
        return concept, target


def concept_collate_fn(batch, tokenizer, max_length: int = 32):
    """Collates concepts into tokenized batch and stacked targets."""
    concepts, targets = zip(*batch)
    encoded = tokenizer(
        list(concepts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    targets = torch.stack(targets, dim=0)
    return encoded["input_ids"], encoded["attention_mask"], targets


class VectorTripleDataset(Dataset):
    """Dataset of (Head Vector, Relation ID, Tail Vector) triples (Stage 1B)."""
    def __init__(self,
                 triples_data: List[Dict],
                 teacher_embeddings: torch.Tensor,
                 concept_to_idx: Dict[str, int],
                 relation_to_idx: Dict[str, int]):
        self.samples = []
        for item in triples_data:
            h = item.get("head")
            r = item.get("relation")
            t = item.get("tail")

            if h in concept_to_idx and t in concept_to_idx and r in relation_to_idx:
                h_idx = concept_to_idx[h]
                t_idx = concept_to_idx[t]
                r_idx = relation_to_idx[r]
                self.samples.append((h_idx, r_idx, t_idx))

        self.teacher_embeddings = teacher_embeddings

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h_idx, r_idx, t_idx = self.samples[idx]
        h_vec = self.teacher_embeddings[h_idx]
        t_vec = self.teacher_embeddings[t_idx]
        return h_vec, r_idx, t_vec


def vector_triple_collate_fn(batch):
    """Collates vector triples into stacked tensors."""
    h_vecs, r_ids, t_vecs = zip(*batch)
    return (
        torch.stack(h_vecs, dim=0),
        torch.tensor(r_ids, dtype=torch.long),
        torch.stack(t_vecs, dim=0)
    )
