import time
from typing import Dict, Any, Optional
from .base import MetricCalculator, ModelContext


class DatasetCodeCalculator(MetricCalculator):
    """Fast calculator for Dataset and Code Score (DAC) metric using metadata only."""
    
    def __init__(self):
        super().__init__("DatasetCode")
    
    def calculate_score(self, context: ModelContext) -> float:
        start_time = time.time()
        
        try:
            if not context:
                score = 0.0
            else:
                has_dataset = self._check_dataset_availability(context)
                has_code = self._check_code_availability(context)
                
                if has_dataset and has_code:
                    score = 1.0
                elif has_dataset or has_code:
                    score = 0.5
                else:
                    score = 0.0
        except Exception as e:
            print(f"Error calculating DAC score: {e}")
            score = 0.5

        end_time = time.time()
        calculation_time_ms = int((end_time - start_time) * 1000)
        self._set_score(score, calculation_time_ms)
        
        return score

    def _check_dataset_availability(self, context: ModelContext) -> bool:
        """Check dataset availability using metadata only."""
        # Check if dataset URL is provided
        if context.dataset_url and context.dataset_url.strip():
            return True
        
        # Check Hugging Face metadata
        if context.huggingface_metadata:
            datasets = context.huggingface_metadata.get('datasets', [])
            if datasets:
                return True
                
            card_data = context.huggingface_metadata.get('cardData', {})
            if 'datasets' in card_data:
                return True
        
        # Check model info
        if context.model_info:
            return 'datasets' in context.model_info or 'dataset' in str(context.model_info).lower()
        
        return False
    
    def _check_code_availability(self, context: ModelContext) -> bool:
        """Check code availability using metadata only."""
        # Check if code URL is provided
        if context.code_url and context.code_url.strip():
            return True
        
        # Check Hugging Face metadata
        if context.huggingface_metadata:
            if 'repository' in context.huggingface_metadata:
                return True
            
            tags = context.huggingface_metadata.get('tags', [])
            code_tags = ['code', 'github', 'source', 'implementation']
            if any(tag.lower() in code_tags for tag in tags if isinstance(tag, str)):
                return True
        
        # Check model info
        if context.model_info and context.model_info.get('source') == 'github':
            return True
        
        return False