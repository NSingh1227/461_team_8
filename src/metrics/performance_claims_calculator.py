from typing import Any, Dict, Optional
import time
import requests
from urllib.parse import urlparse
from .base import MetricCalculator, ModelContext
from ..core.llm_client import ask_for_json_score


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
        except Exception:
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
            if model_id and not model_id.startswith("datasets/"):
                readme_url = f"https://huggingface.co/{model_id}/raw/main/README.md"
                try:
                    resp = requests.get(readme_url, timeout=10)
                    if resp.status_code == 200 and isinstance(resp.text, str):
                        content = resp.text
                        # Combine heuristics + LLM JSON score
                        heuristic = self._heuristic_readme_score(content.lower())
                        prompt = (
                            "Evaluate performance claims in this README.\n"
                            "Rate 0..1 based on standardized benchmarks, citations/links, and reproducibility.\n"
                            "Return {\"score\": float, \"rationale\": string}.\n\n"
                            f"README (first 4000 chars):\n{content[:4000]}"
                        )
                        llm_score, _ = ask_for_json_score(prompt)
                        if llm_score is None:
                            return heuristic
                        return max(0.0, min(1.0, 0.6 * llm_score + 0.4 * heuristic))
                except Exception:
                    pass

        prompt = (
            "Evaluate performance claims in README or model card (0..1).\n"
            "Return {\"score\": float, \"rationale\": string}.\n\n"
            f"URL: {url}"
        )
        llm_score, _ = ask_for_json_score(prompt)
        if llm_score is None:
            return 0.5
        return llm_score

    def _heuristic_readme_score(self, content: str) -> float:
        score = 0.0
        # Evidence of claims: benchmarks, datasets, metrics, citations
        if any(term in content for term in ["benchmark", "leaderboard", "sota", "glue", "superglue", "mmlu"]):
            score += 0.4
        if any(term in content for term in ["accuracy", "f1", "bleu", "rouge", "perplexity", "exact match"]):
            score += 0.3
        if any(term in content for term in ["citation", "arxiv", "doi", "paper"]):
            score += 0.2
        if "evaluation" in content or "results" in content:
            score += 0.1
        return max(0.0, min(1.0, score))


