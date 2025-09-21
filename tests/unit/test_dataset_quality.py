import unittest
import sys
import os

# Add the project root to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.url_processor import process_url, get_handler
from src.metrics.dataset_quality_calculator import DatasetQualityCalculator
from src.metrics.base import ModelContext


class TestURLProcessorWithDQ10(unittest.TestCase):
    def setUp(self):
        self.calc = DatasetQualityCalculator()

    def _process_single_url(self, url: str) -> ModelContext:
        url_type = process_url(url)
        handler = get_handler(url_type)
        if not handler:
            raise ValueError(f"No handler for URL: {url}")
        context = handler.process_url(url)
        if not isinstance(context, ModelContext):
            raise TypeError(f"Handler did not return ModelContext for {url}")
        return context

    def test_datasets_varied(self):
        datasets = {
            "SQuAD": "https://huggingface.co/datasets/squad",
            "HuggingFaceFW": "https://huggingface.co/datasets/HuggingFaceFW/finepdfs",
            "Omniworld": "https://huggingface.co/datasets/InternRobotics/OmniWorld",
            "WikiText-103": "https://huggingface.co/datasets/wikitext",
            "IMDB Reviews": "https://huggingface.co/datasets/imdb",
            "CIFAR-10": "https://huggingface.co/datasets/cifar10",
            "AG News": "https://huggingface.co/datasets/ag_news",
            "Yelp Reviews": "https://huggingface.co/datasets/yelp_review_full",
            "Tiny Shakespeare": "https://huggingface.co/datasets/karpathy/tiny_shakespeare",
            "Toy Text": "https://huggingface.co/datasets/lhoestq/toy_text_dataset",
            "English Random": "https://huggingface.co/datasets/Julien/english_random",
            "Dummy Noise": "https://huggingface.co/datasets/Julien/random_noise"
        }

        for label, url in datasets.items():
            with self.subTest(dataset=label):
                context = self._process_single_url(url)
                score = self.calc.calculate_score(context)
                print(f"[DatasetQuality] {label} ({url}) → score={score:.2f}")
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
