import time
from typing import Dict, Any, Optional
from .base import MetricCalculator, ModelContext
from ..core.config import Config
import requests
from ..core.http_client import get_with_rate_limit, head_with_rate_limit
from ..core.rate_limiter import APIService

class DatasetCodeCalculator(MetricCalculator):
    """Calculator for Dataset and Code Score (DAC) metric."""
    
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
        if context.dataset_url:
            return self._verify_url_accessible(context.dataset_url)
        
        if context.huggingface_metadata:
            datasets = context.huggingface_metadata.get('datasets', [])
            if datasets:
                return True
                
            card_data = context.huggingface_metadata.get('cardData', {})
            if 'datasets' in card_data:
                return True
        
        if context.model_info:
            return 'datasets' in context.model_info or 'dataset' in str(context.model_info).lower()
        
        return False
    
    def _check_code_availability(self, context: ModelContext) -> bool:
        if context.code_url:
            return self._verify_url_accessible(context.code_url)
        
        if context.huggingface_metadata:
            if 'repository' in context.huggingface_metadata:
                return True
            
            tags = context.huggingface_metadata.get('tags', [])
            code_tags = ['code', 'github', 'source', 'implementation']
            if any(tag.lower() in code_tags for tag in tags if isinstance(tag, str)):
                return True
        
        if context.model_info and context.model_info.get('source') == 'github':
            return True
        
        return False
    
    def _verify_url_accessible(self, url: str) -> bool:
        try:
            response = head_with_rate_limit(url, APIService.GENERAL_HTTP, timeout=5, allow_redirects=True)
            if response and response.status_code == 200:
                return True
        except Exception:
            pass
        
        try:
            response = get_with_rate_limit(url, APIService.GENERAL_HTTP, timeout=5, stream=True)
            return response and response.status_code == 200
        except Exception:
            return False