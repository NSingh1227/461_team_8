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
            datasets: Any = context.huggingface_metadata.get('datasets', [])
            if datasets:
                return True
                
            card_data: Dict[str, Any] = context.huggingface_metadata.get('cardData', {})
            if 'datasets' in card_data:
                return True
        
        if context.model_info:
            return 'datasets' in context.model_info or 'dataset' in str(context.model_info).lower()
        
        return False
    
    def _check_code_availability(self, context: ModelContext) -> bool:
        if context.code_url and context.code_url.strip():
            return True
        
        if context.huggingface_metadata:
            if 'repository' in context.huggingface_metadata:
                return True
            
            tags: Any = context.huggingface_metadata.get('tags', [])
            code_tags: List[str] = ['code', 'github', 'source', 'implementation']
            if any(tag.lower() in code_tags for tag in tags if isinstance(tag, str)):
                return True
        
        if context.model_info and context.model_info.get('source') == 'github':
            return True
        
        return False
