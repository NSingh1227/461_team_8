"""
using Halstead complexity measures
"""

from typing import Tuple, List
import time
import os
import tempfile
from radon.complexity import cc_visit
from radon.metrics import h_visit
from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError


class RampUpTimeCalculator:
    
    def calculate_ramp_up_time(self, model_url: str) -> Tuple[float, int]:
        
        start_time = time.time()
        
        try:
            repo_id = self._extract_repo_id(model_url)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=temp_dir,
                    repo_type="model"
                )
                
                python_files = self._find_python_files(temp_dir)
                
                if not python_files:
                    score = 0.5 
                else:
                    avg_difficulty = self._calculate_average_difficulty(python_files)
                    score = self._difficulty_to_ramp_up_score(avg_difficulty)
                    
        except Exception as e:
            print(f"Error processing {model_url}: {e}")
            score = 0.5
        
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        return score, latency_ms
    
    def _find_python_files(self, directory: str) -> List[str]:
        python_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        return python_files
    
    def _calculate_average_difficulty(self, python_files: List[str]) -> float:
        total_difficulty = 0.0
        valid_files = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                halstead_metrics = h_visit(code)
                
                if halstead_metrics and hasattr(halstead_metrics, 'difficulty'):
                    total_difficulty += halstead_metrics.difficulty
                    valid_files += 1
                    
            except Exception as e:
                continue
        
        if valid_files == 0:
            return 10.0  
        
        return total_difficulty / valid_files
    
    def _difficulty_to_ramp_up_score(self, difficulty: float) -> float:
        """
        Convert Halstead difficulty to ramp-up score (0.0-1.0).
        """
        if difficulty <= 5.0:
            return 1.0 
        elif difficulty <= 10.0:
            return 0.8
        elif difficulty <= 20.0:
            return 0.6  
        elif difficulty <= 30.0:
            return 0.4  
        else:
            return 0.2 
        
    def _extract_repo_id(self, model_url: str) -> str:
        if "huggingface.co/" in model_url:
            parts = model_url.split("huggingface.co/")[-1]
            repo_id = parts.split("/tree/").split("/blob/")
            return repo_id
        else:
            raise ValueError(f"Invalid URL: {model_url}")
