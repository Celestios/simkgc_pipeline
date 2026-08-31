import unittest
import json
import struct
import tempfile
import numpy as np
from pathlib import Path
from src.demo_inference import CentrodeGraphEngine

class TestCentrodeInference(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.exports_dir = Path(self.temp_dir.name)

        # 1. Create mock concepts dictionary and binary
        concepts = ["Tehran", "Iran", "Paris", "France", "City"]
        dim = 256
        with open(self.exports_dir / "concepts_dict.json", "w", encoding="utf-8") as f:
            json.dump({"num_concepts": len(concepts), "dim": dim, "concepts": concepts}, f)

        # Create normalized random vectors and save as INT8 binary
        np.random.seed(42)
        vectors = np.random.randn(len(concepts), dim).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        int8_vecs = np.clip(np.round(vectors * 127.0), -127, 127).astype(np.int8)

        header = struct.pack("<4sIII", b"CKGE", len(concepts), dim, 1)
        with open(self.exports_dir / "concepts_256d_int8.bin", "wb") as f:
            f.write(header)
            f.write(int8_vecs.tobytes())

        # 2. Create mock relations ontology and vectors
        relations = {
            "PartOf": {"en_label": "part of", "fa_label": "بخشی_از"},
            "IsA": {"en_label": "is a", "fa_label": "نوعی_از"}
        }
        with open(self.exports_dir / "relations_ontology.json", "w", encoding="utf-8") as f:
            json.dump(relations, f)

        rel_names = sorted(list(relations.keys()))
        with open(self.exports_dir / "relations_metadata.json", "w", encoding="utf-8") as f:
            json.dump({"num_relations": len(rel_names), "dim": dim, "relations": rel_names}, f)

        rel_vecs = np.random.randn(len(rel_names), dim).astype(np.float32)
        rel_vecs = rel_vecs / np.linalg.norm(rel_vecs, axis=1, keepdims=True)
        int8_rel = np.clip(np.round(rel_vecs * 127.0), -127, 127).astype(np.int8)
        rel_header = struct.pack("<4sIII", b"CKGE", len(rel_names), dim, 1)
        with open(self.exports_dir / "relations_256d_int8.bin", "wb") as f:
            f.write(rel_header)
            f.write(int8_rel.tobytes())

        self.engine = CentrodeGraphEngine(self.exports_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_engine_initialization(self):
        self.assertEqual(len(self.engine.concepts), 5)
        self.assertEqual(self.engine.dim, 256)
        self.assertEqual(len(self.engine.rel_names), 2)

    def test_suggest_next_nodes(self):
        suggestions = self.engine.suggest_next_nodes("Tehran", "PartOf", top_k=3)
        self.assertGreaterEqual(len(suggestions), 1)
        # Verify self-loop exclusion (Tehran not in suggestions)
        for name, score in suggestions:
            self.assertNotEqual(name, "Tehran")
            self.assertIsInstance(score, float)

    def test_suggest_relation_between_nodes(self):
        rel_suggestions = self.engine.suggest_relation_between_nodes("Tehran", "Iran", top_k=2)
        self.assertEqual(len(rel_suggestions), 2)
        self.assertIn(rel_suggestions[0][0], ["IsA", "PartOf"])

    def test_discover_concept_tags(self):
        tags = self.engine.discover_concept_tags("Paris", top_k=3)
        self.assertGreaterEqual(len(tags), 1)
        for name, score in tags:
            self.assertNotEqual(name, "Paris")

    def test_audit_connection_sanity(self):
        report = self.engine.audit_connection_sanity("Tehran", "PartOf", "Iran")
        self.assertEqual(report["head"], "Tehran")
        self.assertEqual(report["relation"], "PartOf")
        self.assertEqual(report["tail"], "Iran")
        self.assertIn(report["verdict"], ["LOGICAL", "SUSPICIOUS / LOW ALIGNMENT"])

if __name__ == "__main__":
    unittest.main()
