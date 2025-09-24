from typing import Optional, Dict, Any
import sys
import time
from .base import MetricCalculator, ModelContext


class CodeQualityCalculator(MetricCalculator):

    def __init__(self) -> None:
        super().__init__("CodeQuality")

    def calculate_score(self, context: ModelContext) -> float:
        start_time = time.time()
        
        try:

            if context.model_info and 'github_metadata' in context.model_info:
                score = self._score_from_github_metadata(context.model_info['github_metadata'])
            else:

                score = self._score_from_hf_metadata(context)
        except Exception as e:
            print(f"CodeQuality calculation error: {e}", file=sys.stderr)
            score = 0.5

        end_time = time.time()
        self._set_score(score, int((end_time - start_time) * 1000))
        return max(0.0, min(1.0, score))

    def _score_from_github_metadata(self, github_data: Dict[str, Any]) -> float:
        score = 0.3
        

        language = github_data.get('language', '').lower()
        if language in ['python', 'jupyter notebook']:
            score += 0.2
        

        stars = github_data.get('stargazers_count', 0)
        if stars > 1000:
            score += 0.3
        elif stars > 100:
            score += 0.2
        elif stars > 10:
            score += 0.1
        

        updated_at = github_data.get('updated_at', '')
        if updated_at:
            if '2024' in updated_at or '2025' in updated_at:
                score += 0.1
        

        if github_data.get('description'):
            score += 0.1
        

        if not github_data.get('archived', False):
            score += 0.1
        

        if github_data.get('topics'):
            score += 0.05
        
        return min(1.0, score)

    def _score_from_hf_metadata(self, context: ModelContext) -> float:
        score = 0.4
        
        if context.huggingface_metadata:

            downloads = context.huggingface_metadata.get('downloads', 0)
            if downloads > 1000000:
                score += 0.3
            elif downloads > 100000:
                score += 0.2
            elif downloads > 10000:
                score += 0.1
            

            likes = context.huggingface_metadata.get('likes', 0)
            if likes > 100:
                score += 0.2
            elif likes > 10:
                score += 0.1
            

            if context.huggingface_metadata.get('tags'):
                score += 0.1
        
        return min(1.0, score)
