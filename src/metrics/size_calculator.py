from typing import Dict, List, Optional
import os
import json
import time
import tempfile
from urllib.parse import urlparse
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from .base import MetricCalculator, ModelContext


class SizeCalculator(MetricCalculator):


    HARDWARE_CAPS_MB: Dict[str, int] = {
        "raspberry_pi": 200,
        "jetson_nano": 1000,
        "desktop_pc": 8192,
        "aws_server": 51200,
    }


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

        self.platform_compatibility: Dict[str, float] = {}

    def calculate_score(self, context: ModelContext) -> float:
        start_time = time.time()
        try:
            total_artifact_size_mb = self._estimate_artifact_size_mb(context)

            if total_artifact_size_mb is None:

                self.platform_compatibility = {}
                score = 0.5
            else:

                self.platform_compatibility = {
                    platform: max(0.0, 1.0 - (total_artifact_size_mb / cap_mb))
                    for platform, cap_mb in self.HARDWARE_CAPS_MB.items()
                }
                score = max(self.platform_compatibility.values()) if self.platform_compatibility else 0.5
        except Exception:
            score = 0.5

        end_time = time.time()
        self._set_score(score, int((end_time - start_time) * 1000))
        return score

    def get_platform_compatibility(self) -> Dict[str, float]:
        return self.platform_compatibility

    def get_platform_compatibility_json(self) -> str:
        return json.dumps(self.platform_compatibility, sort_keys=True)

    def _estimate_artifact_size_mb(self, context: ModelContext) -> Optional[float]:
        url = getattr(context, "model_url", "") or ""
        parsed = urlparse(url)
        if parsed.netloc == "huggingface.co":
            repo_id = parsed.path.strip("/")

            if "/tree/" in repo_id:
                repo_id = repo_id.split("/tree/")[0]
            if "/blob/" in repo_id:
                repo_id = repo_id.split("/blob/")[0]
            
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


            total_bytes = 0
            if isinstance(siblings, list):
                for s in siblings:
                    filename = getattr(s, "rfilename", "") or (s.get("rfilename") if isinstance(s, dict) else "")
                    size_bytes = getattr(s, "size", None) or (s.get("size") if isinstance(s, dict) else None)
                    
                    if filename and self._looks_like_artifact(str(filename)) and size_bytes:
                        total_bytes += int(size_bytes)


            if total_bytes > 0:
                return total_bytes / (1024 * 1024)
            

            return self._estimate_size_from_model_type(repo_id)
            
        except (HfHubHTTPError, RepositoryNotFoundError, Exception):
            return None

    def _estimate_size_from_model_type(self, repo_id: str) -> Optional[float]:
        try:
            api = HfApi()
            info = api.model_info(repo_id)
            

            config = getattr(info, "config", None)
            if config:

                if "num_parameters" in config:
                    num_params = config["num_parameters"]

                    estimated_bytes = num_params * 4
                    return estimated_bytes / (1024 * 1024)
            

            repo_lower = repo_id.lower()
            if "tiny" in repo_lower or "small" in repo_lower:
                return 50
            elif "base" in repo_lower or "medium" in repo_lower:
                return 500
            elif "large" in repo_lower:
                return 2000
            elif "xl" in repo_lower or "xxl" in repo_lower:
                return 5000
            else:
                return 1000
                
        except Exception:
            return 1000

    

