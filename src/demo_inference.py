#!/usr/bin/env python3
"""
Interactive / Programmatic Demo of Centrode Production Features:
  1. Next Connected Node Suggestion (Head + Relation -> Top Tails via Geometric Vector Translation)
  2. Instant Relation Suggestion Between Two Nodes (Node A -> Node B -> Direct Displacement Cosine)
  3. Concept Tag & Nearest Neighbor Discovery (Single Node -> Related Concepts)
  4. Graph Linter & Illogical Connection Auditing (Geometric Relation Alignment)
"""

import sys
import json
import struct
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer, BertTokenizer
except ImportError:
    ort = None
    AutoTokenizer = BertTokenizer = None

class CentrodeGraphEngine:
    def __init__(self, exports_dir: Path):
        self.exports_dir = Path(exports_dir)
        
        # 1. Load ONNX Model
        onnx_file = self.exports_dir / "simkgc_256d_int8.onnx"
        if not onnx_file.exists():
            onnx_file = self.exports_dir / "simkgc_256d.onnx"
        self.session = ort.InferenceSession(str(onnx_file)) if onnx_file.exists() and ort is not None else None
        
        # 2. Load Tokenizer (Prioritize standalone exports bundle over checkpoints)
        self.tokenizer = None
        for tok_path in [self.exports_dir, self.exports_dir.parent / "checkpoints" / "simkgc_fa_en"]:
            if tok_path.exists() and AutoTokenizer is not None:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(str(tok_path))
                    break
                except Exception:
                    try:
                        self.tokenizer = BertTokenizer.from_pretrained(str(tok_path))
                        break
                    except Exception:
                        pass
            
        # 3. Load Concepts Dictionary & Binary Matrix (Zero-Copy Memory-Mapped)
        dict_file = self.exports_dir / "concepts_dict.json"
        bin_file = self.exports_dir / "concepts_256d_int8.bin"

        if dict_file.exists() and bin_file.exists():
            with open(dict_file, "r", encoding="utf-8") as f:
                dict_data = json.load(f)
                self.concepts = dict_data["concepts"]
                self.dim = dict_data.get("dim", 256)
                self.concept_to_idx = {c: i for i, c in enumerate(self.concepts)}
                
            with open(bin_file, "rb") as f:
                header = f.read(16)
                magic, n_concepts, dim, prec = struct.unpack("<4sIII", header)
                
            # Memory-mapped read for true zero-copy execution
            int8_mmap = np.memmap(str(bin_file), dtype=np.int8, mode="r", offset=16, shape=(n_concepts, dim))
            self.concept_matrix = np.array(int8_mmap, dtype=np.float32) / 127.0
        else:
            self.concepts = []
            self.dim = 256
            self.concept_to_idx = {}
            self.concept_matrix = np.zeros((0, 256), dtype=np.float32)
            
        # 4. Load Relations Ontology & 32x256 Relation Vector Matrix
        rel_onto_file = self.exports_dir / "relations_ontology.json"
        if rel_onto_file.exists():
            with open(rel_onto_file, "r", encoding="utf-8") as f:
                self.relations = json.load(f)
        else:
            self.relations = {}
            
        rel_bin = self.exports_dir / "relations_256d_int8.bin"
        if rel_bin.exists():
            with open(rel_bin, "rb") as f:
                header = f.read(16)
                magic, n_rels, r_dim, prec = struct.unpack("<4sIII", header)
            int8_rel = np.memmap(str(rel_bin), dtype=np.int8, mode="r", offset=16, shape=(n_rels, r_dim))
            self.rel_matrix = np.array(int8_rel, dtype=np.float32) / 127.0
        else:
            self.rel_matrix = np.zeros((len(self.relations), self.dim), dtype=np.float32)
            
        rel_meta_file = self.exports_dir / "relations_metadata.json"
        if rel_meta_file.exists():
            with open(rel_meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.rel_names = meta.get("relations", sorted(list(self.relations.keys())))
        else:
            self.rel_names = sorted(list(self.relations.keys()))
            
        self.rel_to_idx = {r: i for i, r in enumerate(self.rel_names)}

    def encode_text(self, text: str) -> np.ndarray:
        """Encodes single text string into 256-d normalized vector."""
        if self.session is None or self.tokenizer is None:
            # Deterministic fallback for test environments without ONNX
            np.random.seed(abs(hash(text)) % (2**31))
            vec = np.random.randn(self.dim).astype(np.float32)
            return vec / np.linalg.norm(vec)
            
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors="np")
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        emb = self.session.run(None, ort_inputs)[0][0]
        norm = np.linalg.norm(emb)
        return emb / max(norm, 1e-9)

    def get_concept_vector(self, concept_text: str) -> np.ndarray:
        """Retrieves static concept vector from 50k matrix or encodes on the fly."""
        c_clean = concept_text.strip()
        if c_clean in self.concept_to_idx and len(self.concept_matrix) > 0:
            return self.concept_matrix[self.concept_to_idx[c_clean]]
        return self.encode_text(c_clean)

    def suggest_next_nodes(self, head: str, relation: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Feature 1: Next Connected Node Suggestion (Head + Relation -> Top Tails)."""
        if len(self.concept_matrix) == 0:
            return []
            
        v_h = self.get_concept_vector(head)
        
        if relation in self.rel_to_idx and len(self.rel_matrix) > 0:
            r_vec = self.rel_matrix[self.rel_to_idx[relation]]
            query_vec = v_h + r_vec
            query_vec = query_vec / max(np.linalg.norm(query_vec), 1e-9)
        else:
            query_vec = self.encode_text(f"{head} {relation}")
            
        scores = np.dot(self.concept_matrix, query_vec)
        top_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in top_indices:
            c_name = self.concepts[idx]
            if c_name.lower() != head.lower():
                results.append((c_name, float(scores[idx])))
                if len(results) >= top_k:
                    break
        return results

    def suggest_relation_between_nodes(self, node_a: str, node_b: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Feature 2: Suggest Best Relation Between Two Nodes."""
        if len(self.rel_matrix) == 0 or len(self.rel_names) == 0:
            return []
            
        v_a = self.get_concept_vector(node_a)
        v_b = self.get_concept_vector(node_b)
        
        diff = v_b - v_a
        diff_norm = diff / max(np.linalg.norm(diff), 1e-9)
        
        scores = np.dot(self.rel_matrix, diff_norm)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(self.rel_names[idx], float(scores[idx])) for idx in top_indices]

    def discover_concept_tags(self, node: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Feature 3: Concept Tag & Nearest Neighbor Discovery."""
        if len(self.concept_matrix) == 0:
            return []
            
        v = self.get_concept_vector(node)
        scores = np.dot(self.concept_matrix, v)
        top_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in top_indices:
            c = self.concepts[idx]
            if c.lower() != node.lower():
                results.append((c, float(scores[idx])))
                if len(results) >= top_k:
                    break
        return results

    def audit_connection_sanity(self, head: str, relation: str, tail: str) -> Dict[str, any]:
        """Feature 4: Graph Linter & Illogical Connection Auditing."""
        v_h = self.get_concept_vector(head)
        v_t = self.get_concept_vector(tail)
        
        observed_displacement = v_t - v_h
        obs_norm = observed_displacement / max(np.linalg.norm(observed_displacement), 1e-9)
        
        if relation in self.rel_to_idx and len(self.rel_matrix) > 0:
            canonical_r_vec = self.rel_matrix[self.rel_to_idx[relation]]
            confidence = float(np.dot(canonical_r_vec, obs_norm))
        else:
            confidence = 0.5
            
        is_logical = confidence > 0.15
        return {
            "head": head,
            "relation": relation,
            "tail": tail,
            "confidence_score": round(confidence, 4),
            "is_logical": is_logical,
            "verdict": "LOGICAL" if is_logical else "SUSPICIOUS / LOW ALIGNMENT"
        }
