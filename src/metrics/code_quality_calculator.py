import sys
import time
from typing import Any, Dict, List

from ..core.model_analyzer import ModelDynamicAnalyzer
from .base import MetricCalculator, ModelContext


class CodeQualityCalculator(MetricCalculator):

    def __init__(self) -> None:
        super().__init__("CodeQuality")

    def calculate_score(self, context: ModelContext) -> float:
        start_time: float = time.time()
        
        try:
            score: float
            if context.model_info and 'github_metadata' in context.model_info:
                score = self._score_from_github_metadata(context.model_info['github_metadata'])
            elif context.model_url and 'huggingface.co' in context.model_url:
                score = self._score_from_dynamic_analysis(context.model_url)
            else:
                score = self._score_from_hf_metadata(context)
        except Exception as e:
            print(f"CodeQuality calculation error: {e}", file=sys.stderr)
            score = 0.5

        end_time: float = time.time()
        self._set_score(score, int((end_time - start_time) * 1000))
        return max(0.0, min(1.0, score))

    def _score_from_github_metadata(self, github_data: Dict[str, Any]) -> float:
        score: float = 0.3
        
        language: str = github_data.get('language', '').lower()
        if language in ['python', 'jupyter notebook']:
            score += 0.2
        
        stars: int = github_data.get('stargazers_count', 0)
        if stars > 1000:
            score += 0.3
        elif stars > 100:
            score += 0.2
        elif stars > 10:
            score += 0.1
        
        updated_at: str = github_data.get('updated_at', '')
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
        score: float = 0.4
        
        if context.huggingface_metadata:
            downloads: int = context.huggingface_metadata.get('downloads', 0)
            if downloads > 1000000:
                score += 0.3
            elif downloads > 100000:
                score += 0.2
            elif downloads > 10000:
                score += 0.1
            
            likes: int = context.huggingface_metadata.get('likes', 0)
            if likes > 100:
                score += 0.2
            elif likes > 10:
                score += 0.1
            
            if context.huggingface_metadata.get('tags'):
                score += 0.1
        
        return min(1.0, score)
    
    def _score_from_dynamic_analysis(self, model_url: str) -> float:
        """Score based on dynamic model analysis."""
        try:
            from urllib.parse import urlparse
            
            # Extract repo ID from URL
            parsed_url = urlparse(model_url)
            repo_id: str = parsed_url.path.strip("/")
            
            if "/tree/" in repo_id:
                repo_id = repo_id.split("/tree/")[0]
            if "/blob/" in repo_id:
                repo_id = repo_id.split("/blob/")[0]
            
            if not repo_id:
                return 0.5
            
            # Use dynamic analyzer
            analyzer: ModelDynamicAnalyzer = ModelDynamicAnalyzer()
            try:
                analysis: Dict[str, Any] = analyzer.analyze_model_loading(repo_id)
                validation: Dict[str, Any] = analyzer.validate_model_completeness(repo_id)
                
                score: float = 0.3  # Base score
                
                # Score based on loading capabilities
                if analysis.get("can_load_model", False):
                    score += 0.3
                if analysis.get("can_load_tokenizer", False):
                    score += 0.2
                
                # Score based on completeness
                completeness_score: float = validation.get("completeness_score", 0.0)
                score += completeness_score * 0.2
                
                # Score based on test scripts
                test_score: float = self._check_test_scripts(repo_id)
                score += test_score * 0.1
                
                return min(1.0, score)
                
            finally:
                analyzer.cleanup()
                
        except Exception as e:
            print(f"Dynamic analysis failed for {model_url}: {e}", file=sys.stderr)
            return 0.5
    
    def _check_test_scripts(self, repo_id: str) -> float:
        """Check for test scripts and sample notebooks."""
        try:
            from huggingface_hub import HfApi
            
            api: HfApi = HfApi()
            repo_files: List[Dict[str, Any]] = api.list_repo_files(repo_id=repo_id, repo_type="model")  # type: ignore
            
            test_files: List[str] = []
            notebook_files: List[str] = []
            
            for file_info in repo_files:
                filename: str = file_info.get("path", "")
                if filename.lower().endswith('.py') and 'test' in filename.lower():
                    test_files.append(filename)
                elif filename.lower().endswith(('.ipynb', '.notebook')):
                    notebook_files.append(filename)
            
            score: float = 0.0
            
            # Score for test files
            if test_files:
                score += 0.5
                # Bonus for multiple test files
                if len(test_files) > 1:
                    score += 0.2
            
            # Score for sample notebooks
            if notebook_files:
                score += 0.3
                # Bonus for multiple notebooks
                if len(notebook_files) > 1:
                    score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            print(f"Test script check failed for {repo_id}: {e}", file=sys.stderr)
            return 0.0
