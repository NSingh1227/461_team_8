import sys
import time
from typing import Any, Dict, List, Optional

from .base import MetricCalculator, ModelContext
from .llm_analyzer import LLMAnalyzer


class DatasetQualityCalculator(MetricCalculator):

    def __init__(self) -> None:
        super().__init__("DatasetQuality")
        self.llm_analyzer: LLMAnalyzer = LLMAnalyzer()

    def calculate_score(self, context: ModelContext) -> float:
        start_time: float = time.time()
        score: float = 0.0

        try:
            dataset_info: Optional[Dict[str, Any]] = self._prepare_dataset_info(context)
            if dataset_info:
                score = self.llm_analyzer.analyze_dataset_quality(dataset_info)
            else:
                print("[DatasetQuality] No dataset info available → default 0.0", file=sys.stderr)
        except Exception as e:
            print(f"[DatasetQuality] Error calculating score: {e}", file=sys.stderr)
            score = 0.0

        end_time: float = time.time()
        self._set_score(score, int((end_time - start_time) * 1000))
        return score

    def _prepare_dataset_info(self, context: ModelContext) -> Optional[Dict[str, Any]]:
        dataset_info: Dict[str, Any] = {}

        if context and context.dataset_url:
            dataset_info["dataset_url"] = context.dataset_url

        if context and context.huggingface_metadata:
            # Ensure huggingface_metadata is a dictionary
            if not isinstance(context.huggingface_metadata, dict):
                print(f"DatasetQuality: huggingface_metadata is not a dictionary: {type(context.huggingface_metadata)}", file=sys.stderr)
                return dataset_info or None
                
            datasets: Any = context.huggingface_metadata.get("datasets", [])
            if datasets:
                dataset_info["datasets"] = datasets

            card_data: Dict[str, Any] = context.huggingface_metadata.get("cardData", {})
            if card_data:
                dataset_info.update(card_data)

        readme_content: Optional[str] = self._fetch_readme_content(context)
        if readme_content:
            dataset_info["readme"] = readme_content

        return dataset_info or None

    def _fetch_readme_content(self, context: ModelContext) -> Optional[str]:
        try:
            if not context or not context.huggingface_metadata:
                return None

            # Ensure huggingface_metadata is a dictionary
            if not isinstance(context.huggingface_metadata, dict):
                print(f"DatasetQuality: huggingface_metadata is not a dictionary in _fetch_readme_content: {type(context.huggingface_metadata)}", file=sys.stderr)
                return None

            readme_parts: List[str] = []
            card_data: Dict[str, Any] = context.huggingface_metadata.get("cardData", {})

            if "description" in card_data:
                readme_parts.append(f"# Description\n{card_data['description']}")

            datasets: Any = context.huggingface_metadata.get("datasets", [])
            if datasets:
                readme_parts.append(f"## Datasets\nThis model uses: {', '.join(datasets)}")

            return "\n\n".join(readme_parts) if readme_parts else None
        except Exception as e:
            print(f"[DatasetQuality] Error building README: {e}", file=sys.stderr)
            return None
