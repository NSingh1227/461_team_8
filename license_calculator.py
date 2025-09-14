"""
LicenseCalculator Module for evaluating compatibility with LGPL v2.1
"""

from typing import Dict, Tuple
import time
import re
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

class LicenseCalculator:
    # Hardcoded compatible / non-compatible Licenses
    LGPL_license_compatibility: Dict[str, float] = {
        # Compatible with LGPL
        'mit': 1,
        'apache': 1.0,
        'apache 2.0': 1.0,
        'apache-2.0': 1.0,
        'bsd': 1.0,
        'bsd-3-clause': 1.0,
        'bsd-2-clause': 1.0,
        'lgpl': 1.0,
        'lgpl-2.1': 1.0,
        'lgpl-3.0': 1.0,
        'public domain': 1.0,
        'unlicense': 1.0,
        
        # Incompatible with LGPL
        'gpl': 0.0,
        'gpl-2.0': 0.0,
        'gpl-3.0': 0.0,
        'agpl': 0.0,
        'agpl-3.0': 0.0,
        'gemma': 0.0,
        'proprietary': 0.0,
        'commercial': 0.0,
        'all rights reserved': 0.0,
    }

    def __init__(self) -> None:
        """Intitialize LicenseCalculator"""
        self.hf_api = HfApi()
    
    def calculate_license_score(self, model_url:str) -> Tuple[float, int]:
        """
        Args: 
            readme_content: Full README.md from hugging face
        
        Returns:
            Tuple:
                score: 0.0 OR 0.5 OR 1.0 compatibility score
                latency_ms: Time taken to compute (ms)
        """

        start_time = time.time()
        
        try:
            repo_id = self._extract_repo_id(model_url)
            readme_content = self._fetch_readme_from_hf_api(repo_id)
            score = self._calculate_compatibility_score(readme_content)

        except Exception as e:
            print(f"Error processing {model_url}: {e}")
            score = 0.5
        
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        return (score, latency_ms)

    def _extract_repo_id(self, model_url: str) -> str:
        """Extract repo id from Hugging face url"""
        if "huggingface.co/" in model_url:
            repo_id = "/".join(model_url.split("huggingface.co/")[1].split("/"))
            
            return repo_id
        else:
            raise ValueError(f"Invalid Hugging Face URL: {model_url}")

    def _fetch_readme_from_hf_api(self, repo_id: str) -> str:
        """Fetch README.md content using Hugging face API"""
        try:
            readme_path = hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type="model",
                cache_dir=None
            )

            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return content
        
        except (RepositoryNotFoundError, HfHubHTTPError) as e:
            print(f"Could not fetch README content for {repo_id}: {e}")
            return ""
    
    def _calculate_compatibility_score(self, readme_content:str) -> float:
        """Calculate compatibility score of license"""
        if not readme_content: return 0.5

        license_text = self._extract_license_from_readme(readme_content)

        if not license_text: return 0.5

        for known_license, score in self.LGPL_license_compatibility.items():
            if known_license == license_text: return score
        
        return 0.5

    def _extract_license_from_readme(self, readme_content: str) -> str:
        """Extract license section from README content"""
        license_pattern = r'license:\s*([^\n]*)'
        match = re.search(license_pattern, readme_content)

        if match: 
            license = match.group(1).lower().strip()
            return license
        return ""