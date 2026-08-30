import json
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset

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
    Collates raw string triplets into tokenized PyTorch tensors for head-relation and tail.
    Optionally attaches teacher concept target indices for zero-overhead distillation.
    """
    
    def __init__(self, tokenizer, max_seq_length: int = 64, concept_to_idx: Optional[Dict[str, int]] = None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.concept_to_idx = concept_to_idx
        self.sep_token = tokenizer.sep_token if tokenizer.sep_token else "[SEP]"

    def __call__(self, batch: List[Tuple[str, str, str]]) -> Dict[str, torch.Tensor]:
        hr_texts = [f"{head} {self.sep_token} {relation}" for head, relation, _ in batch]
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
            tail_indices = [self.concept_to_idx.get(tail, 0) for tail in tail_texts]
            batch_dict["tail_indices"] = torch.tensor(tail_indices, dtype=torch.long)
            
        return batch_dict
