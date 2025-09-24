from typing import Optional, Dict, Any
import time
from .base import MetricCalculator, ModelContext


class CodeQualityCalculator(MetricCalculator):
    """
    Fast code quality scoring using GitHub API metadata only.
    No cloning, no LLM calls - just API-based heuristics for speed.
    """

    def __init__(self) -> None:
        super().__init__("CodeQuality")

    def calculate_score(self, context: ModelContext) -> float:
        start_time = time.time()
        
        try:
            # Use GitHub metadata if available
            if context.model_info and 'github_metadata' in context.model_info:
                score = self._score_from_github_metadata(context.model_info['github_metadata'])
            else:
                # Fallback to Hugging Face metadata
                score = self._score_from_hf_metadata(context)
        except Exception as e:
            print(f"CodeQuality calculation error: {e}")
            score = 0.5

        end_time = time.time()
        self._set_score(score, int((end_time - start_time) * 1000))
        return max(0.0, min(1.0, score))

    def _score_from_github_metadata(self, github_data: Dict[str, Any]) -> float:
        """Fast scoring based on GitHub API metadata only."""
        score = 0.3  # Base score
        
        # Language indicator (Python is good for ML)
        language = github_data.get('language', '').lower()
        if language in ['python', 'jupyter notebook']:
            score += 0.2
        
        # Stars indicate community trust
        stars = github_data.get('stargazers_count', 0)
        if stars > 1000:
            score += 0.3
        elif stars > 100:
            score += 0.2
        elif stars > 10:
            score += 0.1
        
        # Recent activity
        updated_at = github_data.get('updated_at', '')
        if updated_at:
            if '2024' in updated_at or '2025' in updated_at:
                score += 0.1
        
        # Has description
        if github_data.get('description'):
            score += 0.1
        
        # Not archived
        if not github_data.get('archived', False):
            score += 0.1
        
        # Has topics/tags
        if github_data.get('topics'):
            score += 0.05
        
        return min(1.0, score)

    def _score_from_hf_metadata(self, context: ModelContext) -> float:
        """Fast scoring based on Hugging Face metadata only."""
        score = 0.4  # Base score for HF models
        
        if context.huggingface_metadata:
            # Downloads indicate popularity
            downloads = context.huggingface_metadata.get('downloads', 0)
            if downloads > 1000000:
                score += 0.3
            elif downloads > 100000:
                score += 0.2
            elif downloads > 10000:
                score += 0.1
            
            # Likes indicate quality
            likes = context.huggingface_metadata.get('likes', 0)
            if likes > 100:
                score += 0.2
            elif likes > 10:
                score += 0.1
            
            # Has tags
            if context.huggingface_metadata.get('tags'):
                score += 0.1
        
        return min(1.0, score)