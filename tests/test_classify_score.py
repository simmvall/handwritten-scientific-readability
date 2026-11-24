import unittest

from src.document_readability import classify_score


class TestClassifyScore(unittest.TestCase):
    def test_equal_thirds_boundaries(self):
        self.assertEqual(classify_score(1.0, "baseline_consistency"), "High")
        self.assertEqual(classify_score(0.67, "baseline_consistency"), "High")
        self.assertEqual(classify_score(0.669, "spatial_organization"), "Moderate")
        self.assertEqual(classify_score(0.34, "spatial_organization"), "Moderate")
        self.assertEqual(classify_score(0.339, "combined_score"), "Low")
        self.assertEqual(classify_score(0.0, "combined_score"), "Low")

    def test_fragmentation_labels(self):
        self.assertEqual(
            classify_score(0.70, "text_fragmentation"),
            "Low Fragmentation (High Readability)",
        )
        self.assertEqual(classify_score(0.50, "text_fragmentation"), "Moderate Fragmentation")
        self.assertEqual(
            classify_score(0.20, "text_fragmentation"),
            "High Fragmentation (Low Readability)",
        )


if __name__ == "__main__":
    unittest.main()
