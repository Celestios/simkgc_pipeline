#!/usr/bin/env python3
"""
Interactive / Programmatic Demo of Centrode Production Features:
  1. Next Connected Node Suggestion (Head + Relation -> Top Tails)
  2. Best Relation Suggestion Between Two Nodes (Node A -> Node B -> Best Relation)
  3. Concept Tag & Nearest Neighbor Discovery (Single Node -> Related Concepts)
  4. Graph Linter & Illogical Connection Auditing (Connection Sanity Score)
"""

import json
import struct
import numpy as np
import onnxruntime as ort
from pathlib import Path
from transformers import AutoTokenizer, BertTokenizer

class CentrodeGraphEngine:
    def __init__(self, exports_dir: Path):
        self.exports_dir = exports_dir
        
        # 1. Load ONNX Model
        onnx_file = exports_dir / "simkgc_256d_int8.onnx"
        if not onnx_file.exists():
            onnx_file = exports_dir / "simkgc_256d.onnx"
        self.session = ort.InferenceSession(str(onnx_file))
        
        # 2. Load Tokenizer
        chk_dir = exports_dir.parent / "checkpoints" / "simkgc_fa_en"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(chk_dir))
        except Exception:
            self.tokenizer = BertTokenizer.from_pretrained(str(chk_dir))
            
        # 3. Load Concepts Dictionary & Binary Matrix
        with open(exports_dir / "concepts_dict.json", "r", encoding="utf-8") as f:
            dict_data = json.load(f)
            self.concepts = dict_data["concepts"]
            self.dim = dict_data["dim"]
            
        with open(exports_dir / "concepts_256d_int8.bin", "rb") as f:
            header = f.read(16)
            magic, n_concepts, dim, prec = struct.unpack("<4sIII", header)
            raw_bytes = f.read()
            # Dequantize INT8 to float unit vectors for search
            int8_matrix = np.frombuffer(raw_bytes, dtype=np.int8).reshape(n_concepts, dim)
            self.concept_matrix = int8_matrix.astype(np.float32) / 127.0
            
        # 4. Load Canonical Relations
        with open(exports_dir / "relations_ontology.json", "r", encoding="utf-8") as f:
            self.relations = json.load(f)

    def encode_text(self, text: str) -> np.ndarray:
        """Encodes single text string into 256-d normalized vector via ONNX."""
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors="np")
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        emb = self.session.run(None, ort_inputs)[0]
        norm = np.linalg.norm(emb, axis=-1, keepdims=True)
        return emb / np.maximum(norm, 1e-9)

    def suggest_next_nodes(self, head: str, relation: str, top_k: int = 5):
        """Feature 1: Predicts top candidate tails for (Head + Relation)."""
        query_text = f"{head} [SEP] {relation}"
        q_vec = self.encode_text(query_text) # [1, 256]
        scores = np.dot(self.concept_matrix, q_vec.T).squeeze()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.concepts[idx], float(scores[idx])) for idx in top_indices]

    def suggest_relation_between_nodes(self, node_a: str, node_b: str, top_k: int = 3):
        """Feature 2: Scans canonical relations to find best fit between Node A and Node B."""
        target_vec = self.encode_text(node_b) # [1, 256]
        rel_scores = []
        for rel_name in self.relations.keys():
            q_text = f"{node_a} [SEP] {rel_name}"
            q_vec = self.encode_text(q_text)
            score = float(np.dot(q_vec, target_vec.T).squeeze())
            rel_scores.append((rel_name, score))
            
        rel_scores.sort(key=lambda x: x[1], reverse=True)
        return rel_scores[:top_k]

    def suggest_tags_and_similar_concepts(self, node: str, top_k: int = 5):
        """Feature 3: Nearest neighbor concept tags for a standalone node."""
        node_vec = self.encode_text(node)
        scores = np.dot(self.concept_matrix, node_vec.T).squeeze()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.concepts[idx], float(scores[idx])) for idx in top_indices if self.concepts[idx] != node]

    def audit_connection_sanity(self, node_a: str, relation: str, node_b: str) -> dict:
        """Feature 4: Graph Linter checking semantic validity of an existing edge."""
        q_vec = self.encode_text(f"{node_a} [SEP] {relation}")
        target_vec = self.encode_text(node_b)
        score = float(np.dot(q_vec, target_vec.T).squeeze())
        
        if score >= 0.70:
            status = "VALID"
            message = "Strong semantic connection."
        elif score >= 0.40:
            status = "WEAK"
            best_alt = self.suggest_relation_between_nodes(node_a, node_b, top_k=1)[0]
            message = f"Weak connection. Consider relation: '{best_alt[0]}' (Score: {best_alt[1]:.2f})."
        else:
            status = "ILLOGICAL"
            message = "Unrelated concepts or contradictory relation."
            
        return {"score": score, "status": status, "message": message}

if __name__ == "__main__":
    exports_path = Path("exports")
    if (exports_path / "concepts_256d_int8.bin").exists():
        engine = CentrodeGraphEngine(exports_path)
        print("Centrode Graph Engine Loaded Successfully.")
