import unittest
import struct
import numpy as np
from pathlib import Path
from src.export import export_concepts_to_rust_binary

class TestExportBinary(unittest.TestCase):
    def test_rust_binary_serializer_and_header(self):
        concepts = ["ایران", "آسیا", "زمین", "جاذبه"]
        num_concepts = len(concepts)
        dim = 256
        vecs = np.random.randn(num_concepts, dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        out_bin = Path("exports/test_concepts.bin")
        out_dict = Path("exports/test_dict.json")

        export_concepts_to_rust_binary(concepts, vecs, out_bin, out_dict, quantize_int8=True)

        self.assertTrue(out_bin.exists())
        self.assertTrue(out_dict.exists())

        # Verify CKGE header
        with open(out_bin, "rb") as f:
            header_bytes = f.read(16)
            magic, n_concepts, d, prec = struct.unpack("<4sIII", header_bytes)
            self.assertEqual(magic, b"CKGE")
            self.assertEqual(n_concepts, num_concepts)
            self.assertEqual(d, dim)
            self.assertEqual(prec, 1) # INT8 code

            payload = f.read()
            self.assertEqual(len(payload), num_concepts * dim)

        # Cleanup test files
        if out_bin.exists():
            out_bin.unlink()
        if out_dict.exists():
            out_dict.unlink()

    def test_concept_filtering_and_selection(self):
        import json
        from src.export import is_persian_text, is_valid_concept_string, select_top_production_concepts
        
        self.assertTrue(is_persian_text("ایران"))
        self.assertTrue(is_persian_text("هوش مصنوعی"))
        self.assertFalse(is_persian_text("Artificial Intelligence"))
        
        self.assertTrue(is_valid_concept_string("دانشگاه تهران"))
        self.assertFalse(is_valid_concept_string("http://example.com"))
        self.assertFalse(is_valid_concept_string("12345"))
        self.assertFalse(is_valid_concept_string("this is an excessively long concept phrase that exceeds maximum word limit"))
        
        dummy_triples = [
            {"head": "ایران", "relation": "IsA", "tail": "کشور", "weight": 2.0},
            {"head": "ایران", "relation": "PartOf", "tail": "آسیا", "weight": 1.5},
            {"head": "تهران", "relation": "IsA", "tail": "پایتخت", "weight": 1.0},
            {"head": "England", "relation": "IsA", "tail": "country", "weight": 2.0},
            {"head": "London", "relation": "IsA", "tail": "capital", "weight": 1.0},
            {"head": "England", "relation": "PartOf", "tail": "Europe", "weight": 1.5},
        ]
        test_json = Path("exports/test_triples.json")
        test_json.parent.mkdir(parents=True, exist_ok=True)
        with open(test_json, "w", encoding="utf-8") as f:
            json.dump(dummy_triples, f)
            
        selected = select_top_production_concepts([str(test_json)], total_quota=4, fa_quota=2, en_quota=2)
        self.assertEqual(len(selected), 4)
        
        if test_json.exists():
            test_json.unlink()

if __name__ == "__main__":
    unittest.main()
