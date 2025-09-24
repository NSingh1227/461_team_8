from typing import Any, Dict, Optional
import time
import requests
from urllib.parse import urlparse
from .base import MetricCalculator, ModelContext

# Import the LLM client from the correct location
try:
    from ..core.llm_client import ask_for_json_score
except ImportError:
    # Fallback if llm_client doesn't exist
    def ask_for_json_score(prompt):
        """Fallback function if LLM client is not available."""
        return 0.5, "LLM client not available"


class PerformanceClaimsCalculator(MetricCalculator):
    """
    Scores how well performance claims are substantiated in README/metadata.
    Heuristics via Hugging Face model card metadata when available:
    - Presence of metrics/results fields or benchmark-related tags increases score
    - Otherwise, fallback to a dummy LLM assessment with the card data
    """

    def __init__(self) -> None:
        super().__init__("PerformanceClaims")

    def calculate_score(self, context: ModelContext) -> float:
        start_time = time.time()
        try:
            score = self._score_from_metadata_or_llm(context)
            # Ensure score is a valid float
            if score is None:
                score = 0.5
            score = float(score)
        except Exception as e:
            print(f"Error in PerformanceClaimsCalculator: {e}")
            score = 0.5

        end_time = time.time()
        self._set_score(score, int((end_time - start_time) * 1000))
        return score

    def _score_from_metadata_or_llm(self, context: ModelContext) -> float:
        # Prefer reading README directly from HF raw endpoint when URL is an HF model
        url = getattr(context, "model_url", "") or ""
        parsed = urlparse(url)
        
        if parsed.netloc == "huggingface.co":
            # Path like /org/name or /datasets/name
            model_id = parsed.path.strip("/")
            # Remove /tree/main or similar git refs from the path
            if "/tree/" in model_id:
                model_id = model_id.split("/tree/")[0]
            if "/blob/" in model_id:
                model_id = model_id.split("/blob/")[0]
            
            if model_id and not model_id.startswith("datasets/"):
                print("model_id: ", model_id)
                readme_url = f"https://huggingface.co/{model_id}/raw/main/README.md"
                print("readme_url: ", readme_url)
                
                try:
                    resp = requests.get(readme_url, timeout=10)
                    print("resp: ", resp)
                    
                    if resp.status_code == 200 and isinstance(resp.text, str):
                        content = resp.text
                        print("content: ", content[:200] + "..." if len(content) > 200 else content)
                        
                        # Combine heuristics + LLM JSON score
                        heuristic = self._heuristic_readme_score(content.lower())
                        
                        prompt = (
                            "Evaluate performance claims in this README.\n"
                            "Rate 0..1 based on standardized benchmarks, citations/links, and reproducibility.\n"
                            "Return {\"score\": float, \"rationale\": string}.\n\n"
                            f"README (first 4000 chars):\n{content[:4000]}"
                        )
                        
                        try:
                            llm_score, _ = ask_for_json_score(prompt)
                            if llm_score is not None and isinstance(llm_score, (int, float)):
                                return max(0.0, min(1.0, 0.6 * llm_score + 0.4 * heuristic))
                            else:
                                return heuristic
                        except Exception as e:
                            print(f"LLM scoring failed: {e}")
                            return heuristic
                    else:
                        print(f"Failed to fetch README: status {resp.status_code}")
                        return 0.3
                        
                except Exception as e:
                    print("Exception: ", e)
                    return 0.3
        else:
            print("Not an HF model")
            return 0.3

    def _heuristic_readme_score(self, content: str) -> float:
        """Heuristic scoring based on README content."""
        score = 0.0
        
        # Evidence of claims: benchmarks, datasets, metrics, citations
        benchmark_terms = ["benchmark", "leaderboard", "sota", "glue", "superglue", "mmlu"]
        if any(term in content for term in benchmark_terms):
            score += 0.4
        
        # Performance metrics mentioned
        metric_terms = ["accuracy", "f1", "bleu", "rouge", "perplexity", "exact match"]
        if any(term in content for term in metric_terms):
            score += 0.3
        
        # Citations and papers
        citation_terms = ["citation", "arxiv", "doi", "paper"]
        if any(term in content for term in citation_terms):
            score += 0.2
        
        # Evaluation sections
        if "evaluation" in content or "results" in content:
            score += 0.1
        
        return max(0.0, min(1.0, score))