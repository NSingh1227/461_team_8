from typing import Optional, Dict, Any
import os
import time
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlparse
from huggingface_hub import hf_hub_download
from ..core.llm_client import ask_for_json_score
from .base import MetricCalculator, ModelContext


class CodeQualityCalculator(MetricCalculator):
    """
    Heuristic code quality scoring.

    Primary signals for GitHub repos (if available via URL processing):
    - Stars (normalized)
    - Forks (normalized)
    - Description presence
    - Primary language presence
    - Recent activity (updated within last 12 months)

    If GitHub signals are not available, fall back to a dummy LLM-based
    assessment using the Hugging Face model card metadata (when available).
    """

    def __init__(self) -> None:
        super().__init__("CodeQuality")

    def calculate_score(self, context: ModelContext) -> float:
        start_time = time.time()
        try:
            code_url = getattr(context, "code_url", None)
            if isinstance(code_url, str) and code_url.strip():
                # Use the GitHub repository for local analysis + metadata
                repo_url = code_url
                model_info = context.model_info if context and context.model_info else {}
                score = self._score_via_clone_and_heuristics(repo_url, model_info)
            else:
                # No code repo provided; use the Hugging Face model URL and analyze local files
                model_url = getattr(context, "model_url", "") or ""
                score = self._score_via_hf_downloads(model_url)
        except Exception:
            # Fall back to GitHub metadata-only heuristics if clone or inputs fail
            try:
                model_info = context.model_info if context and context.model_info else {}
                score = self._score_from_github_metadata(model_info)
            except Exception:
                score = 0.5

        end_time = time.time()
        self._set_score(score, int((end_time - start_time) * 1000))
        return score

    def _score_from_github_metadata(self, model_info: Dict[str, Any]) -> float:
        stars = int(model_info.get("stars") or 0)
        forks = int(model_info.get("forks") or 0)
        description = model_info.get("description")
        updated_at_iso = model_info.get("updated_at")

        score = 0.0

        # Popularity signals
        score += min(1.0, stars / 1000.0) * 0.4
        score += min(1.0, forks / 500.0) * 0.2

        
        if isinstance(description, str) and description.strip():
            score += 0.1

        # Recent activity
        try:
            if isinstance(updated_at_iso, str) and updated_at_iso:
                updated_dt = datetime.fromisoformat(updated_at_iso.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                months = (now.year - updated_dt.year) * 12 + (now.month - updated_dt.month)
                if months <= 12:
                    score += 0.2
        except Exception:
            pass

        return max(0.0, min(1.0, score))

    def _score_via_clone_and_heuristics(self, repo_url: str, model_info: Dict[str, Any]) -> float:
        """
        Clone the repo (pure Python Git lib), inspect presence of common files,
        count simple signals, and blend with GitHub metadata as available.
        """
        gh_score = self._score_from_github_metadata(model_info)

        try:
            from dulwich import porcelain  # type: ignore
        except Exception:
            return gh_score  # fall back to metadata-only

        signals = 0.0
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                porcelain.clone(source=repo_url, target=tmpdir, checkout=True)
                # Presence of structure/config/test files (including scripts dir)
                candidates = [
                    "README.md", "README.rst", "pyproject.toml", "setup.py",
                    "requirements.txt", "config.json", "model_index.json",
                    "tests", "test", "scripts", ".flake8", "mypy.ini", "setup.cfg",
                ]
                found = 0
                for c in candidates:
                    path = os.path.join(tmpdir, c)
                    if os.path.isdir(path) or os.path.isfile(path):
                        found += 1
                if found > 0:
                    # normalize by count (cap influence)
                    signals += min(1.0, found / 7.0) * 0.5

                # README style/comprehensiveness via heuristic + LLM
                readme_text = None
                for fname in ["README.md", "README.rst"]:
                    p = os.path.join(tmpdir, fname)
                    if os.path.exists(p):
                        try:
                            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                                readme_text = f.read()
                            break
                        except Exception:
                            pass
                heuristic_readme = 0.0
                if isinstance(readme_text, str):
                    heuristic_readme = self._analyze_readme_content_score(readme_text)
                llm_score = None
                if isinstance(readme_text, str) and len(readme_text) > 0:
                    prompt = (
                        "Rate README quality and maintainability (0..1).\n"
                        "Consider clarity, examples, contribution/testing docs, code style hints, badges.\n"
                        "Return {\"score\": float, \"rationale\": string}.\n\n"
                        f"README:\n{readme_text[:4000]}"
                    )
                    llm_score, _ = ask_for_json_score(prompt)
                if llm_score is None:
                    llm_score = heuristic_readme
                signals += min(1.0, max(heuristic_readme, llm_score)) * 0.3

        except Exception:
            # If clone fails, return the metadata-based score
            return gh_score

        return max(0.0, min(1.0, gh_score * 0.6 + signals))

    def _score_via_hf_downloads(self, model_url: str) -> float:
        """
        Download HF model files (README.md, config.json, model_index.json),
        analyze README style/comprehensiveness heuristically plus a dummy LLM
        score for config/structure, and blend into a final score.
        """
        try:
            parsed = urlparse(model_url)
            if parsed.netloc != "huggingface.co":
                return 0.6

            repo_id = parsed.path.strip("/")
            if not repo_id or repo_id.startswith("datasets/"):
                return 0.6

            import tempfile
            readme_text = None
            config_present = False
            index_present = False

            with tempfile.TemporaryDirectory() as tmpdir:
                # Try to fetch README.md
                try:
                    readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="model", cache_dir=tmpdir)
                    with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                        readme_text = f.read()
                except Exception:
                    pass

                # Try to fetch config.json / model_index.json
                for fname in ["config.json", "model_index.json"]:
                    try:
                        local_path = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="model", cache_dir=tmpdir)
                        if os.path.exists(local_path):
                            if fname == "config.json":
                                config_present = True
                            else:
                                index_present = True
                    except Exception:
                        continue

            readme_score = 0.0
            if isinstance(readme_text, str):
                readme_score = self._analyze_readme_content_score(readme_text)

            # LLM assessment using HF model files
            prompt = (
                "Rate maintainability (0..1) using HF files.\n"
                "Consider architecture clarity, parameters, example scripts, structure.\n"
                "Return {\"score\": float, \"rationale\": string}.\n\n"
                f"config.json present: {config_present}\n"
                f"model_index.json present: {index_present}\n"
                f"README (first 4000 chars):\n{(readme_text or '')[:4000]}"
            )
            structure_score, _ = ask_for_json_score(prompt)
            if structure_score is None:
                structure_score = 0.6 if (config_present or index_present) else 0.55

            final = max(0.0, min(1.0, 0.6 * readme_score + 0.4 * structure_score))
            # Ensure stable baseline when little is available
            if final == 0.0:
                final = 0.6
            return final
        except Exception:
            return 0.6

    def _analyze_readme_content_score(self, content: str) -> float:
        content_l = content.lower()
        score = 0.0
        # Headings / sections
        if "install" in content_l or "installation" in content_l:
            score += 0.2
        if "usage" in content_l or "getting started" in content_l:
            score += 0.2
        if "example" in content_l or "examples" in content_l:
            score += 0.15
        if "contribut" in content_l:
            score += 0.1
        if "license" in content_l:
            score += 0.05
        if "test" in content_l:
            score += 0.05
        # Code blocks
        if "```" in content or "\n    " in content:  # code fence or indented code
            score += 0.15
     
        return max(0.0, min(1.0, score))


