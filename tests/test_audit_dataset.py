import unittest
import json
import tempfile
from pathlib import Path
from src.data.audit_dataset import audit_dataset

class TestAuditDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_path = Path(self.temp_dir.name) / "test_triples.json"
        self.clean_out_path = Path(self.temp_dir.name) / "clean_triples.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empty_dataset_safety(self):
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump([], f)

        res = audit_dataset(self.data_path)
        self.assertEqual(res["total_triples"], 0)
        self.assertEqual(res["conflicts"], 0)
        self.assertEqual(res["asymmetric_loops"], 0)

    def test_contradiction_detection(self):
        triples = [
            {"head": "نور", "relation": "Synonym", "tail": "روشنایی", "lang": "fa"},
            {"head": "نور", "relation": "Antonym", "tail": "روشنایی", "lang": "fa"},  # Contradiction
            {"head": "Earth", "relation": "IsA", "tail": "Planet", "lang": "en"},
        ]
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(triples, f)

        res = audit_dataset(self.data_path)
        self.assertEqual(res["total_triples"], 3)
        self.assertEqual(res["conflicts"], 1)

    def test_asymmetric_cycle_detection(self):
        triples = [
            {"head": "باران", "relation": "Causes", "tail": "سیل", "lang": "fa"},
            {"head": "سیل", "relation": "Causes", "tail": "باران", "lang": "fa"},  # Asymmetric loop
        ]
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(triples, f)

        res = audit_dataset(self.data_path)
        self.assertEqual(res["asymmetric_loops"], 1)

    def test_script_anomaly_and_clean_export(self):
        triples = [
            {"head": "دانشگاه ملّي", "relation": "IsA", "tail": "دانشگاه", "lang": "fa"},  # Arabic Yeh
            {"head": "Car", "relation": "UsedFor", "tail": "http://example.com", "lang": "en"},  # URL
        ]
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(triples, f)

        res = audit_dataset(self.data_path, output_clean_path=self.clean_out_path)
        self.assertEqual(res["script_anomalies"], 2)
        self.assertTrue(self.clean_out_path.exists())

        with open(self.clean_out_path, "r", encoding="utf-8") as f:
            cleaned = json.load(f)
        self.assertEqual(len(cleaned), 2)
        self.assertIn("ی", cleaned[0]["head"])
        self.assertNotIn("ي", cleaned[0]["head"])

if __name__ == "__main__":
    unittest.main()
