import unittest
from src.data.cleaner import normalize_text, clean_knowledge_graph

class TestDataCleaner(unittest.TestCase):
    def test_persian_normalization(self):
        # Arabic Yeh/Kaf conversion
        raw_text = "دانشگاه ملّي كشور ايران"
        normalized = normalize_text(raw_text, "fa")
        self.assertIn("ی", normalized)
        self.assertIn("ک", normalized)
        self.assertNotIn("ي", normalized)
        self.assertNotIn("ك", normalized)

    def test_suffix_stripping(self):
        # ConceptNet tag removal
        raw = "car (noun)"
        normalized = normalize_text(raw, "en")
        self.assertEqual(normalized, "car")

    def test_self_loop_and_duplicate_removal(self):
        raw_triples = [
            {"head": "ایران", "relation": "RelatedTo", "tail": "ایران", "weight": 1.0}, # Self loop
            {"head": "زمین", "relation": "HasProperty", "tail": "جاذبه", "weight": 1.0},
            {"head": "زمین", "relation": "HasProperty", "tail": "جاذبه", "weight": 2.5}, # Duplicate with higher weight
            {"head": "a", "relation": "b", "tail": "c", "weight": 0.1}, # Below min weight
        ]
        cleaned = clean_knowledge_graph(raw_triples, min_weight=1.0)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["head"], "زمین")
        self.assertEqual(cleaned[0]["weight"], 2.5)

if __name__ == "__main__":
    unittest.main()
