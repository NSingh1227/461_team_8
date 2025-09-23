import time
import os
import re
from typing import Dict, Any, Optional
from .base import MetricCalculator, ModelContext
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError


class SizeCalculator(MetricCalculator):
    """Calculator for model size compatibility across hardware platforms."""
    
    # Hardware platform memory limits in MB
    HARDWARE_LIMITS = {
        'raspberry_pi': 200,
        'jetson_nano': 1000, 
        'desktop_pc': 8192,
        'aws_server': 51200
    }
    
    def __init__(self):
        super().__init__("Size")
    
    def calculate_score(self, context: ModelContext) -> Dict[str, float]:
        """Calculate size compatibility scores for each hardware platform."""
        start_time = time.time()
        
        try:
            model_size_mb = self._get_model_size(context)
            platform_scores = {}
            
            for platform, limit_mb in self.HARDWARE_LIMITS.items():
                if model_size_mb <= 0:
                    # Unknown size - give moderate compatibility
                    platform_scores[platform] = 0.5
                else:
                    # Calculate compatibility: max(0, 1 - (size / limit))
                    compatibility = max(0.0, 1.0 - (model_size_mb / limit_mb))
                    platform_scores[platform] = round(compatibility, 3)
            
        except Exception as e:
            print(f"Error calculating size score: {e}")
            # Default scores if calculation fails
            platform_scores = {platform: 0.5 for platform in self.HARDWARE_LIMITS.keys()}
        
        end_time = time.time()
        calculation_time_ms = int((end_time - start_time) * 1000)
        
        # Store the dictionary of scores instead of a single score
        self._score = platform_scores
        self._calculation_time_ms = calculation_time_ms
        
        return platform_scores
    
    def _get_model_size(self, context: ModelContext) -> float:
        """Get total model size in MB from various sources."""
        if context.model_url.startswith("https://huggingface.co"):
            return self._get_huggingface_model_size(context)
        elif context.model_url.startswith("https://github.com"):
            return self._get_github_repo_size(context)
        else:
            return 0.0
    
    def _get_huggingface_model_size(self, context: ModelContext) -> float:
        """Get HuggingFace model size by checking model files."""
        try:
            repo_id = self._extract_repo_id(context.model_url)
            if not repo_id:
                return 0.0
            
            # Get list of files in the repository
            files = list_repo_files(repo_id, repo_type="model")
            
            total_size_mb = 0.0
            model_files = [
                'pytorch_model.bin', 'model.safetensors', 'tf_model.h5',
                'model.onnx', 'model.pkl', 'pytorch_model-*.bin'
            ]
            
            for file_path in files:
                filename = os.path.basename(file_path)
                
                # Check if it's a model weight file
                is_model_file = any(
                    filename.startswith(pattern.replace('*', '')) or 
                    filename == pattern
                    for pattern in model_files
                )
                
                if is_model_file:
                    try:
                        # Try to get file info (this may not always work)
                        # For now, estimate based on common model sizes
                        if 'pytorch_model' in filename:
                            # Estimate based on common model architectures
                            total_size_mb += self._estimate_pytorch_size(filename)
                        elif 'safetensors' in filename:
                            total_size_mb += 200  # Typical safetensors size
                        elif 'tf_model' in filename:
                            total_size_mb += 300  # Typical TensorFlow model size
                        else:
                            total_size_mb += 100  # Default estimate
                            
                    except Exception:
                        continue
            
            return total_size_mb if total_size_mb > 0 else 500  # Default estimate
            
        except (RepositoryNotFoundError, HfHubHTTPError) as e:
            print(f"Could not access HuggingFace model files: {e}")
            return 500  # Default estimate
        except Exception as e:
            print(f"Error getting HuggingFace model size: {e}")
            return 0.0
    
    def _estimate_pytorch_size(self, filename: str) -> float:
        """Estimate PyTorch model size based on filename patterns."""
        # This is a simplified estimation - in practice you'd want to 
        # actually download and check file sizes
        if 'large' in filename.lower():
            return 1000  # 1GB estimate for large models
        elif 'base' in filename.lower():
            return 500   # 500MB estimate for base models
        elif 'small' in filename.lower() or 'tiny' in filename.lower():
            return 100   # 100MB estimate for small models
        else:
            return 300   # Default 300MB estimate
    
    def _get_github_repo_size(self, context: ModelContext) -> float:
        """Estimate GitHub repository size."""
        try:
            # For GitHub repos, we can't easily get model file sizes
            # Use metadata if available
            if (context.model_info and 
                'github_metadata' in context.model_info):
                
                github_data = context.model_info['github_metadata']
                repo_size_kb = github_data.get('size', 0)  # Size in KB
                return repo_size_kb / 1024.0  # Convert to MB
            
            return 100  # Default estimate for GitHub repos
            
        except Exception as e:
            print(f"Error getting GitHub repo size: {e}")
            return 0.0
    
    def _extract_repo_id(self, model_url: str) -> Optional[str]:
        """Extract repository ID from HuggingFace URL."""
        try:
            if "huggingface.co/" in model_url:
                parts = model_url.split("huggingface.co/")[1].split("/")
                if len(parts) >= 2:
                    return f"{parts[0]}/{parts[1]}"
            return None
        except Exception:
            return None