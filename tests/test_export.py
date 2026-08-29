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

if __name__ == "__main__":
    unittest.main()
