import os
import json
import struct
import unittest
import tempfile
import numpy as np
from pathlib import Path
from src.export import (
    is_valid_concept_string,
    select_top_production_concepts,
    export_concepts_to_rust_binary,
    export_relations_metadata,
    write_ckge_binary,
    quantize_int8_matrix
)

class TestExportBinary(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name)
        self.bin_path = self.out_dir / "test_concepts.bin"
        self.dict_path = self.out_dir / "test_dict.json"
        self.meta_path = self.out_dir / "test_relations.json"
        self.triples_path = self.out_dir / "test_triples.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_ckge_binary_serialization(self):
        num_concepts = 100
        dim = 256
        embeddings = np.random.randn(num_concepts, dim).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
        concepts = [f"concept_{i}" for i in range(num_concepts)]

        export_concepts_to_rust_binary(
            concepts=concepts,
            embeddings=embeddings,
            bin_output_path=self.bin_path,
            dict_output_path=self.dict_path,
            quantize_int8=True
        )

        self.assertTrue(self.bin_path.exists())
        self.assertTrue(self.dict_path.exists())

        # Verify Header (16 bytes)
        with open(self.bin_path, "rb") as f:
            header = f.read(16)
            magic, n_c, d, prec = struct.unpack("<4sIII", header)
            self.assertEqual(magic, b"CKGE")
            self.assertEqual(n_c, num_concepts)
            self.assertEqual(d, dim)
            self.assertEqual(prec, 1)

            payload = f.read()
            self.assertEqual(len(payload), num_concepts * dim)

    def test_relations_metadata_export(self):
        export_relations_metadata(self.meta_path)
        self.assertTrue(self.meta_path.exists())
        with open(self.meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("IsA", data)
        self.assertIn("PartOf", data)

    def test_lexical_filters(self):
        self.assertTrue(is_valid_concept_string("هوش مصنوعی"))
        self.assertTrue(is_valid_concept_string("machine learning"))
        self.assertFalse(is_valid_concept_string("a"))  # Too short
        self.assertFalse(is_valid_concept_string("12345"))  # Pure digits
        self.assertFalse(is_valid_concept_string("this is a very long sentence with many words"))  # > 4 words
        self.assertFalse(is_valid_concept_string("http://example.com/item"))  # URL

    def test_concept_curation_quotas(self):
        mock_triples = [
            {"head": "ایران", "relation": "IsA", "tail": "کشور", "weight": 3.0},
            {"head": "تهران", "relation": "PartOf", "tail": "ایران", "weight": 2.0},
            {"head": "Paris", "relation": "IsA", "tail": "City", "weight": 3.0},
            {"head": "France", "relation": "HasPart", "tail": "Paris", "weight": 2.0},
            {"head": "London", "relation": "IsA", "tail": "City", "weight": 2.0},
        ]
        with open(self.triples_path, "w", encoding="utf-8") as f:
            json.dump(mock_triples, f)

        selected = select_top_production_concepts(
            data_paths=[str(self.triples_path)],
            total_quota=4,
            fa_quota=2,
            en_quota=2
        )
        self.assertEqual(len(selected), 4)

if __name__ == "__main__":
    unittest.main()
