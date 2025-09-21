from typing import Dict, Optional
import time
import re
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from .base import MetricCalculator, ModelContext

class LicenseCalculator(MetricCalculator):
    """Calculator for LGPL v2.1 license compatibility scoring."""
    LGPL_license_compatibility: Dict[str, float] = {
        'mit': 1.0,
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
        super().__init__("License")
        self.hf_api = HfApi()
    
    def calculate_score(self, context: ModelContext) -> float:
        start_time = time.time()
        
        try:
            license_text = self._extract_license_from_context(context)
            score = self._calculate_compatibility_score(license_text)
        except Exception as e:
            print(f"Error calculating license score: {e}")
            score = 0.5
        
        end_time = time.time()
        calculation_time_ms = int((end_time - start_time) * 1000)
        self._set_score(score, calculation_time_ms)
        
        return score
    
    def _extract_license_from_context(self, context: ModelContext) -> Optional[str]:
        if context.model_url.startswith("https://huggingface.co"):
            return self._extract_huggingface_license(context)
        elif context.model_url.startswith("https://github.com"):
            return self._extract_github_license(context)
        else:
            return None

    def _extract_huggingface_license(self, context: ModelContext) -> Optional[str]:
        if context.huggingface_metadata:
            if 'cardData' in context.huggingface_metadata:
                card_data = context.huggingface_metadata['cardData']
                if 'license' in card_data:
                    return str(card_data['license']).lower().strip()
            
            if 'tags' in context.huggingface_metadata:
                for tag in context.huggingface_metadata['tags']:
                    if isinstance(tag, str) and 'license:' in tag:
                        return tag.replace('license:', '').strip().lower()
        
        try:
            repo_id = self._extract_repo_id(context.model_url)
            readme_content = self._fetch_readme_from_hf_api(repo_id)
            return self._extract_license_from_readme(readme_content)
        except Exception as e:
            print(f"Failed to fetch README for license: {e}")
            return None

    def _extract_github_license(self, context: ModelContext) -> Optional[str]:
        if (context.model_info and 
            'github_metadata' in context.model_info and 
            context.model_info['github_metadata']):
            
            github_data = context.model_info['github_metadata']
            if 'license' in github_data and github_data['license']:
                license_info = github_data['license']
                if 'spdx_id' in license_info and license_info['spdx_id']:
                    return license_info['spdx_id'].lower().strip()
                elif 'name' in license_info and license_info['name']:
                    return license_info['name'].lower().strip()
        
        return None
    
    def _calculate_compatibility_score(self, license_text: Optional[str]) -> float:
        if not license_text:
            return 0.5

        license_text = license_text.lower().strip()
        
        if license_text in self.LGPL_license_compatibility:
            return self.LGPL_license_compatibility[license_text]
        
        for known_license, score in self.LGPL_license_compatibility.items():
            if known_license in license_text or license_text in known_license:
                return score
        
        return 0.5
    
    def _extract_license_from_readme(self, readme_content: str) -> Optional[str]:
        license_pattern = r'license:\s*([^\n]*)'
        match = re.search(license_pattern, readme_content.lower())

        if match: 
            license = match.group(1).lower().strip()
            return license
        return None

    def _extract_repo_id(self, model_url: str) -> str:
        if "huggingface.co/" in model_url:
            repo_id = "/".join(model_url.split("huggingface.co/")[1].split("/"))
            return repo_id
        else:
            raise ValueError(f"Invalid Hugging Face URL: {model_url}")

    def _fetch_readme_from_hf_api(self, repo_id: str) -> str:
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