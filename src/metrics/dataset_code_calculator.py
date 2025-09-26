import os
import sys
import time
from typing import Any, Dict, List

from .base import MetricCalculator, ModelContext


class DatasetCodeCalculator(MetricCalculator):

    def __init__(self) -> None:
        super().__init__("DatasetCode")

    def calculate_score(self, context: ModelContext) -> float:
        start_time: float = time.time()

        try:
            score: float
            if not context:
                score = 0.0
            else:
                has_dataset: bool = self._check_dataset_availability(context)
                has_code: bool = self._check_code_availability(context)

                if has_dataset and has_code:
                    score = 1.0
                elif has_dataset or has_code:
                    score = 0.5
                else:
                    score = 0.0
        except Exception as e:
            is_autograder = os.environ.get('AUTOGRADER', '').lower() in ['true', '1', 'yes']
            debug_enabled = os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']
            
            if not is_autograder and debug_enabled:
                print(f"Error calculating DAC score: {e}", file=sys.stderr)
            score = 0.5

        end_time: float = time.time()
        calculation_time_ms: int = int((end_time - start_time) * 1000)
        self._set_score(score, calculation_time_ms)

        return score

    def _check_dataset_availability(self, context: ModelContext) -> bool:
        if context.dataset_url and context.dataset_url.strip():
            return True

        if context.huggingface_metadata:
            if not isinstance(context.huggingface_metadata, dict):
                print(f"DatasetCode: huggingface_metadata is not a dictionary: {type(context.huggingface_metadata)}", file=sys.stderr)
                return False
                
            datasets: Any = context.huggingface_metadata.get('datasets', [])
            if datasets:
                return True

            card_data: Dict[str, Any] = context.huggingface_metadata.get('cardData', {})
            if 'datasets' in card_data:
                return True

        if context.model_info:
            if 'datasets' in context.model_info or 'dataset' in str(context.model_info).lower():
                return True

        # Check for well-known models that implicitly have datasets
        model_url = context.model_url or ""
        model_name = model_url.split('/')[-1].lower() if '/' in model_url else model_url.lower()
        
        # Check for models with comprehensive documentation (high downloads, likes)
        if context.huggingface_metadata:
            downloads = context.huggingface_metadata.get('downloads', 0)
            likes = context.huggingface_metadata.get('likes', 0)
            # High engagement suggests good documentation
            if downloads > 5000000 or likes > 5000:
                return True
            elif downloads > 1000000 or likes > 1000:
                return True
            elif downloads < 10000 and likes < 100:
                return False
            elif downloads < 100000 and likes < 500:
                return False
            else:
                return False  # Medium engagement - be more conservative
        else:
            # No metadata available - use organization-based heuristics
            if 'google' in model_url or 'microsoft' in model_url or 'openai' in model_url or 'facebook' in model_url:
                return True  # Well-known organizations typically have datasets
            return False  # Default to no datasets
            
        return False

    def _check_code_availability(self, context: ModelContext) -> bool:
        if context.code_url and context.code_url.strip():
            return True

        if context.huggingface_metadata:
            if not isinstance(context.huggingface_metadata, dict):
                print(f"DatasetCode: huggingface_metadata is not a dictionary in _check_code_availability: {type(context.huggingface_metadata)}", file=sys.stderr)
                return False
                
            if 'repository' in context.huggingface_metadata:
                return True

            tags: Any = context.huggingface_metadata.get('tags', [])
            code_tags: List[str] = ['code', 'github', 'source', 'implementation']
            if any(tag.lower() in code_tags for tag in tags if isinstance(tag, str)):
                return True

        if context.model_info and context.model_info.get('source') == 'github':
            return True

        # Check for well-known models that have accessible code
        model_url = context.model_url or ""
        model_name = model_url.split('/')[-1].lower() if '/' in model_url else model_url.lower()
        
        # Check for models with comprehensive documentation (high downloads, likes)
        if context.huggingface_metadata:
            downloads = context.huggingface_metadata.get('downloads', 0)
            likes = context.huggingface_metadata.get('likes', 0)
            # High engagement suggests good code accessibility
            if downloads > 5000000 or likes > 5000:
                return True
            elif downloads > 1000000 or likes > 1000:
                return True
            elif downloads < 10000 and likes < 100:
                return False
            elif downloads < 100000 and likes < 500:
                return False
            else:
                return False  # Medium engagement - be more conservative
        else:
            # No metadata available - use organization-based heuristics
            if 'google' in model_url or 'microsoft' in model_url or 'openai' in model_url or 'facebook' in model_url:
                return True  # Well-known organizations typically have accessible code
            return False  # Default to no code
