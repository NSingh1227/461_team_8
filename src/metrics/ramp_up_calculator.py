import sys
import time
from typing import Optional
from .base import MetricCalculator, ModelContext
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from urllib.parse import urlparse


class RampUpCalculator(MetricCalculator):
    
    def __init__(self):
        super().__init__("RampUp")
    
    def calculate_score(self, context: ModelContext) -> float:
        start_time = time.time()
        
        try:

            model_url = getattr(context, "model_url", "") or ""
            if "huggingface.co" in model_url:
                score = self._score_huggingface_model(model_url)
            else:

                score = 0.5
        except Exception as e:
            print(f"Error calculating ramp-up score: {e}", file=sys.stderr)
            score = 0.5
        
        end_time = time.time()
        self._set_score(score, int((end_time - start_time) * 1000))
        return score
    
    def _score_huggingface_model(self, model_url: str) -> float:
        try:

            parsed = urlparse(model_url)
            repo_id = parsed.path.strip("/")
            

            if "/tree/" in repo_id:
                repo_id = repo_id.split("/tree/")[0]
            if "/blob/" in repo_id:
                repo_id = repo_id.split("/blob/")[0]
            
            if not repo_id:
                return 0.3
            

            readme_content = None
            try:
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    readme_path = hf_hub_download(
                        repo_id=repo_id,
                        filename="README.md",
                        repo_type="model",
                        cache_dir=tmpdir
                    )
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        readme_content = f.read()
            except (RepositoryNotFoundError, HfHubHTTPError):
                return 0.2
            except Exception:
                return 0.3
            
            if not readme_content:
                return 0.2
            

            score = self._analyze_readme_quality(readme_content)
            return max(0.2, min(1.0, score))
            
        except Exception:
            return 0.3
    
    def _analyze_readme_quality(self, content: str) -> float:
        content_lower = content.lower()
        score = 0.3
        

        if any(term in content_lower for term in ["install", "pip install", "setup"]):
            score += 0.2
        

        if any(term in content_lower for term in ["usage", "example", "how to"]):
            score += 0.2
        

        if "```" in content or "\n    " in content:
            score += 0.15
        

        if any(term in content_lower for term in ["parameters", "api", "configuration"]):
            score += 0.1
        

        if "getting started" in content_lower or "quick start" in content_lower:
            score += 0.05
        
        return score
