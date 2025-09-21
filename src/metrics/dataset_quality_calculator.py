import time
from typing import Dict, Any, Optional
from .base import MetricCalculator, ModelContext
from .llm_analyzer import LLMAnalyzer


class DatasetQualityCalculator(MetricCalculator):
    """Calculator for Dataset Quality (DQ) metric using Purdue GenAI Studio API."""

    def __init__(self):
        super().__init__("DatasetQuality")
        self.llm_analyzer = LLMAnalyzer()

    def calculate_score(self, context: ModelContext) -> float:
        start_time = time.time()
        score = 0.0

        try:
            dataset_info = self._prepare_dataset_info(context)
            if dataset_info:
                score = self.llm_analyzer.analyze_dataset_quality(dataset_info)
            else:
                print("[DatasetQuality] No dataset info available → default 0.0")
        except Exception as e:
            print(f"[DatasetQuality] Error calculating score: {e}")
            score = 0.0

        end_time = time.time()
        self._set_score(score, int((end_time - start_time) * 1000))
        return score

    def _prepare_dataset_info(self, context: ModelContext) -> Optional[Dict[str, Any]]:
        dataset_info: Dict[str, Any] = {}

        if context and context.dataset_url:
            dataset_info["dataset_url"] = context.dataset_url

        if context and context.huggingface_metadata:
            datasets = context.huggingface_metadata.get("datasets", [])
            if datasets:
                dataset_info["datasets"] = datasets

            card_data = context.huggingface_metadata.get("cardData", {})
            if card_data:
                dataset_info.update(card_data)

        readme_content = self._fetch_readme_content(context)
        if readme_content:
            dataset_info["readme"] = readme_content

        return dataset_info or None

    def _fetch_readme_content(self, context: ModelContext) -> Optional[str]:
        try:
            if not context or not context.huggingface_metadata:
                return None

            readme_parts = []
            card_data = context.huggingface_metadata.get("cardData", {})

            if "description" in card_data:
                readme_parts.append(f"# Description\n{card_data['description']}")

            datasets = context.huggingface_metadata.get("datasets", [])
            if datasets:
                readme_parts.append(f"## Datasets\nThis model uses: {', '.join(datasets)}")

            return "\n\n".join(readme_parts) if readme_parts else None
        except Exception as e:
            print(f"[DatasetQuality] Error building README: {e}")
            return None
