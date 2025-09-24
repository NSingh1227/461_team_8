import json
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from .base import MetricCalculator, ModelContext


class SizeCalculator(MetricCalculator):
    """Calculator for model size metric - measures deployability across platforms."""

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
        """Calculate size score based on model artifacts."""
        start_time: float = time.time()
        try:
            total_artifact_size_mb: Optional[float] = self._estimate_artifact_size_mb(context)

            if total_artifact_size_mb is None:
                self.platform_compatibility = {}
                score: float = 0.5
            else:
                self.platform_compatibility = {
                    platform: max(0.0, 1.0 - (total_artifact_size_mb / cap_mb))
                    for platform, cap_mb in self.HARDWARE_CAPS_MB.items()
                }
                score = max(self.platform_compatibility.values()) if self.platform_compatibility else 0.5
        except Exception:
            score = 0.5

        end_time: float = time.time()
        self._set_score(score, int((end_time - start_time) * 1000))
        return score

    def get_platform_compatibility(self) -> Dict[str, float]:
        """Get platform compatibility scores."""
        return self.platform_compatibility

    def _estimate_artifact_size_mb(self, context: ModelContext) -> Optional[float]:
        """Estimate total artifact size in MB."""
        url: str = getattr(context, "model_url", "") or ""
        parsed = urlparse(url)
        if parsed.netloc == "huggingface.co":
            repo_id: str = parsed.path.strip("/")

            if "/tree/" in repo_id:
                repo_id = repo_id.split("/tree/")[0]
            if "/blob/" in repo_id:
                repo_id = repo_id.split("/blob/")[0]
            
            if repo_id:
                return self._hf_total_artifact_size_mb(repo_id)
        return None

    def _looks_like_artifact(self, filename: str) -> bool:
        """Check if filename looks like a model artifact."""
        lower: str = filename.lower()
        for ext in self.ARTIFACT_EXTENSIONS:
            if lower.endswith(ext) or ext in lower:
                return True
        return False

    def _hf_total_artifact_size_mb(self, repo_id: str) -> Optional[float]:
        """Calculate total artifact size for Hugging Face model."""
        try:
            api: HfApi = HfApi()
            info = api.model_info(repo_id)
            siblings = getattr(info, "siblings", None)

            total_bytes: int = 0
            if isinstance(siblings, list):
                for s in siblings:
                    filename: str = str(getattr(s, "rfilename", "") or (s.get("rfilename") if isinstance(s, dict) else ""))
                    size_bytes: Optional[int] = getattr(s, "size", None) or (s.get("size") if isinstance(s, dict) else None)
                    
                    if filename and self._looks_like_artifact(str(filename)) and size_bytes:
                        total_bytes += int(size_bytes)

            if total_bytes > 0:
                return total_bytes / (1024 * 1024)
            
            return self._estimate_size_from_model_type(repo_id)
            
        except (HfHubHTTPError, RepositoryNotFoundError, Exception):
            return None

    def _estimate_size_from_model_type(self, repo_id: str) -> Optional[float]:
        """Estimate size based on model type and parameters."""
        try:
            api: HfApi = HfApi()
            info = api.model_info(repo_id)
            
            config = getattr(info, "config", None)
            if config:
                if "num_parameters" in config:
                    num_params: int = config["num_parameters"]
                    estimated_bytes: int = num_params * 4
                    return estimated_bytes / (1024 * 1024)
            
            repo_lower: str = repo_id.lower()
            if "tiny" in repo_lower or "small" in repo_lower:
                return 50.0
            elif "base" in repo_lower or "medium" in repo_lower:
                return 500.0
            elif "large" in repo_lower:
                return 2000.0
            elif "xl" in repo_lower or "xxl" in repo_lower:
                return 5000.0
            else:
                return 1000.0
                
        except Exception:
            return 1000.0
    
    def _download_and_analyze_config(self, repo_id: str, filename: str) -> Dict[str, Any]:
        """Download and analyze config.json or model_index.json file."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path: str = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="model",
                    cache_dir=tmpdir
                )
                
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data: Dict[str, Any] = json.load(f)
                
                analysis: Dict[str, Any] = {
                    "filename": filename,
                    "has_config": True,
                    "model_type": config_data.get("model_type", "unknown"),
                    "architecture": config_data.get("architectures", ["unknown"]),
                    "hidden_size": config_data.get("hidden_size", 0),
                    "num_layers": config_data.get("num_hidden_layers", 0),
                    "vocab_size": config_data.get("vocab_size", 0),
                    "intermediate_size": config_data.get("intermediate_size", 0),
                    "num_attention_heads": config_data.get("num_attention_heads", 0)
                }
                
                # Estimate model size based on architecture
                if filename == "config.json":
                    estimated_params: int = self._estimate_model_parameters(config_data)
                    analysis["estimated_parameters"] = estimated_params
                    analysis["estimated_size_mb"] = estimated_params * 4 / (1024 * 1024)  # Assuming 4 bytes per parameter
                
                return analysis
                
        except Exception as e:
            print(f"Error analyzing {filename}: {e}", file=sys.stderr)
            return {"filename": filename, "has_config": False, "error": str(e)}
    
    def _estimate_model_parameters(self, config: Dict[str, Any]) -> int:
        """Estimate model parameters based on config.json."""
        try:
            hidden_size: int = config.get("hidden_size", 0)
            num_layers: int = config.get("num_hidden_layers", 0)
            vocab_size: int = config.get("vocab_size", 0)
            intermediate_size: int = config.get("intermediate_size", 0)
            
            if not all([hidden_size, num_layers, vocab_size]):
                return 0
            
            # Rough estimation for transformer models
            # Embedding layer
            embedding_params: int = vocab_size * hidden_size
            
            # Transformer layers
            attention_params: int = num_layers * (4 * hidden_size * hidden_size)  # Q, K, V, O projections
            ffn_params: int = num_layers * (2 * hidden_size * intermediate_size)  # Feed-forward network
            
            # Layer normalization (approximate)
            ln_params: int = num_layers * (2 * hidden_size)  # Layer norm parameters
            
            total_params: int = embedding_params + attention_params + ffn_params + ln_params
            return total_params
            
        except Exception:
            return 0

