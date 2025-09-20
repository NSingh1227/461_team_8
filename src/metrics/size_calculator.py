from typing import Dict, List, Optional
import os
import time
import tempfile
from urllib.parse import urlparse
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from .base import MetricCalculator, ModelContext


class SizeCalculator(MetricCalculator):
    """
    Calculates a model "Size" score in [0, 1] based on the sizes of model
    artifact files. Uses Hugging Face metadata when available to identify
    artifact file sizes and grades fitness against multiple hardware caps.

    Score per platform: max(0, 1 - (model_artifact_size_mb / cap_mb))
    Final score is the max across platform caps to reward portability.
    """

    # Hardware platform caps in megabytes
    HARDWARE_CAPS_MB: Dict[str, int] = {
        "raspberry_pi": 200,
        "jetson_nano": 1000,
        "desktop_pc": 8192,
        "aws_server": 51200,
    }

    # Common model artifact filename patterns/extensions
    ARTIFACT_EXTENSIONS: List[str] = [
        ".bin",
        ".safetensors",
        ".h5",
        ".onnx",
        ".pt",
        ".ckpt",
        "pytorch_model.bin",
        "tf_model.h5",
        "model.onnx",
    ]

    def __init__(self) -> None:
        super().__init__("Size")

    def calculate_score(self, context: ModelContext) -> float:
        start_time = time.time()
        try:
            total_artifact_size_mb = self._estimate_artifact_size_mb(context)

            if total_artifact_size_mb is None:
                score = 0.5
            else:
                per_platform_scores = [
                    max(0.0, 1.0 - (total_artifact_size_mb / cap_mb))
                    for cap_mb in self.HARDWARE_CAPS_MB.values()
                ]
                score = max(per_platform_scores) if per_platform_scores else 0.5
        except Exception:
            score = 0.5

        end_time = time.time()
        self._set_score(score, int((end_time - start_time) * 1000))
        return score

    def _estimate_artifact_size_mb(self, context: ModelContext) -> Optional[float]:
        """
        Estimate total size of model artifacts in megabytes.
        Uses Hugging Face metadata "siblings" when available.
        Returns None if size cannot be determined.
        """
        url = getattr(context, "model_url", "") or ""
        parsed = urlparse(url)
        if parsed.netloc == "huggingface.co":
            repo_id = parsed.path.strip("/")
            if repo_id:
                return self._hf_total_artifact_size_mb(repo_id)
        return None

    def _looks_like_artifact(self, filename: str) -> bool:
        lower = filename.lower()
        for ext in self.ARTIFACT_EXTENSIONS:
            if lower.endswith(ext) or ext in lower:
                return True
        return False

    def _hf_total_artifact_size_mb(self, repo_id: str) -> Optional[float]:
        try:
            api = HfApi()
            info = api.model_info(repo_id)
            siblings = getattr(info, "siblings", None)

            # Collect candidate artifact filenames
            candidate_files: List[str] = []
            if isinstance(siblings, list):
                for s in siblings:
                    filename = getattr(s, "rfilename", "") or (s.get("rfilename") if isinstance(s, dict) else "")
                    if filename and self._looks_like_artifact(str(filename)):
                        candidate_files.append(str(filename))

            # If siblings unavailable or empty, try common names
            if not candidate_files:
                candidate_files = [
                    "pytorch_model.bin",
                    "model.safetensors",
                    "tf_model.h5",
                    "model.onnx",
                ]

            # Download candidates to a temp dir and sum sizes
            import tempfile
            total_bytes = 0
            with tempfile.TemporaryDirectory() as tmpdir:
                for fname in candidate_files:
                    try:
                        local_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=fname,
                            repo_type="model",
                            cache_dir=tmpdir
                        )
                        if os.path.exists(local_path):
                            total_bytes += os.path.getsize(local_path)
                    except Exception:
                        # File not found or other issue; skip to next candidate
                        continue

            if total_bytes > 0:
                return total_bytes / (1024 * 1024)
            return None
        except (HfHubHTTPError, RepositoryNotFoundError, Exception):
            return None

    


