import unittest
from license_calculator import LicenseCalculator

class TestLicenseCalculator(unittest.TestCase):

    def setUp(self):
        self.calculator = LicenseCalculator()

    def test_gemma_license(self):
        # Gemma license is proprietary, should be 0.0
        model_url = "https://huggingface.co/google/gemma-2b"
        score, _ = self.calculator.calculate_license_score(model_url)
        print(f"Gemma license score: {score}")
        self.assertEqual(score, 0.0)

    def test_mit_license(self):
        # MIT license should be 1.0
        model_url = "https://huggingface.co/distilbert/distilbert-base-uncased"
        score, _ = self.calculator.calculate_license_score(model_url)
        print(f"MIT license score: {score}")
        self.assertEqual(score, 1.0)

    def test_apache_2_0_license(self):
        # Apache-2.0 license should be 1.0
        model_url = "https://huggingface.co/bert-base-uncased"
        score, _ = self.calculator.calculate_license_score(model_url)
        print(f"Apache-2.0 license score: {score}")
        self.assertEqual(score, 1.0)

    def test_unknown_license(self):
        # Unknown license, should be 0.5
        model_url = "https://huggingface.co/unknown-repo/unknown-model" # This repo doesn't exist, will cause an exception and return 0.5
        score, _ = self.calculator.calculate_license_score(model_url)
        print(f"Unknown license score: {score}")
        self.assertEqual(score, 0.5)

if __name__ == '__main__':
    unittest.main()