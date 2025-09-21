import unittest
import sys
import os

# Add the project root to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.url_processor import URLProcessor, process_url, get_handler
from src.metrics.dataset_code_calculator import DatasetCodeCalculator
from src.metrics.base import ModelContext

class TestURLProcessorWithDAC(unittest.TestCase):
    def setUp(self):
        self.calc = DatasetCodeCalculator()

    def _process_single_url(self, url: str) -> ModelContext:
        """Helper: categorizes URL, runs through handler, returns ModelContext."""
        url_type = process_url(url)
        handler = get_handler(url_type)
        if not handler:
            raise ValueError(f"No handler for URL: {url}")
        return handler.process_url(url)

    def test_hf_model_dataset_and_code(self):
        model_url = "https://huggingface.co/google/gemma-3-270m"
        dataset_url = "https://huggingface.co/datasets/xlangai/AgentNet"
        code_url = "https://github.com/SkyworkAI/Matrix-Game"

        # Model
        model_context = self._process_single_url(model_url)
        # Attach dataset + code manually (since model handler won't know them automatically)
        model_context.dataset_url = dataset_url
        model_context.code_url = code_url

        score = self.calc.calculate_score(model_context)
        print(f"[HF Model + Dataset + Code] Score = {score}")
        self.assertEqual(score, 1.0)

    def test_only_dataset(self):
        dataset_url = "https://huggingface.co/datasets/xlangai/AgentNet"
        context = self._process_single_url(dataset_url)
        score = self.calc.calculate_score(context)
        print(f"[Dataset only] Score = {score}")
        self.assertEqual(score, 0.5)

    def test_only_code(self):
        code_url = "https://github.com/SkyworkAI/Matrix-Game"
        context = self._process_single_url(code_url)
        score = self.calc.calculate_score(context)
        print(f"[Code only] Score = {score}")
        self.assertEqual(score, 0.5)

    def test_neither(self):
        model_url = "https://huggingface.co/google/gemma-3-270m"
        context = self._process_single_url(model_url)
        score = self.calc.calculate_score(context)
        print(f"[Model (no dataset/code)] Score = {score}")
        self.assertEqual(score, 0.0)

    def test_invalid(self):
        bad_url = "https://notareal.domain/this_should_fail"
        try:
            context = self._process_single_url(bad_url)
        except Exception:
            # If handler fails, build a dummy context
            context = ModelContext(
                model_url=bad_url,
                model_info={"type": "invalid"},
                dataset_url=bad_url,
                code_url=bad_url,
                local_repo_path=None,
                huggingface_metadata=None
            )
        score = self.calc.calculate_score(context)
        print(f"[Invalid URL] Score = {score}")
        self.assertEqual(score, 0.0)  # invalid URLs score as 0.0


if __name__ == "__main__":
    unittest.main()
