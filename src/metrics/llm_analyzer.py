import os
import re
import sys
from typing import Any, Dict, List, Optional

from ..core.http_client import post_with_rate_limit
from ..core.rate_limiter import APIService


class LLMAnalyzer:

    def __init__(self,
                 api_url: str = "https://genai.rcac.purdue.edu/api/chat/completions",
                 model: str = "llama3.1:latest",
                 api_key: Optional[str] = None) -> None:
        self.api_url: str = api_url
        self.model: str = model

        self.api_key: Optional[str] = api_key or os.getenv("GEN_AI_STUDIO_API_KEY")

    def _post_to_genai(self, messages: List[Dict[str, str]]) -> Optional[str]:
        is_autograder: bool = os.environ.get('AUTOGRADER', '').lower() in ['true', '1', 'yes']
        debug_enabled: bool = os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']
        
        if not self.api_key:
            if not is_autograder and debug_enabled:
                print("[LLMAnalyzer] Missing GEN_AI_STUDIO_API_KEY. Please set it in your environment.", file=sys.stderr)
            return None
        try:
            payload: Dict[str, Any] = {"model": self.model, "messages": messages}
            headers: Dict[str, str] = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            response = post_with_rate_limit(
                self.api_url, 
                APIService.GENAI,
                json=payload, 
                headers=headers, 
                timeout=30
            )
            if response and response.status_code == 200:
                data: Dict[str, Any] = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                if response:
                    is_autograder = os.environ.get('AUTOGRADER', '').lower() in ['true', '1', 'yes']
                    debug_enabled = os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']
                    
                    if not is_autograder and debug_enabled:
                        print(f"[LLMAnalyzer] API error {response.status_code}: {response.text}", file=sys.stderr)
                return None
        except Exception as e:
            is_autograder = os.environ.get('AUTOGRADER', '').lower() in ['true', '1', 'yes']
            debug_enabled = os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']
            
            if not is_autograder and debug_enabled:
                print(f"[LLMAnalyzer] Request failed: {e}", file=sys.stderr)
            return None

    def analyze_dataset_quality(self, dataset_info: Dict[str, Any]) -> float:
        user_prompt: str = (
            "Analyze the following dataset information and return ONLY "
            "a numeric quality score between 0.0 and 1.0:\n\n"
            f"{dataset_info}"
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are an evaluator that scores dataset quality."},
            {"role": "user", "content": user_prompt}
        ]
        content: Optional[str] = self._post_to_genai(messages)
        return self._extract_score(content)

    def _extract_score(self, content: Optional[str]) -> float:
        if not content:
            return 0.0
        try:
            match: Optional[re.Match[str]] = re.search(r"-?\d+(?:\.\d+)?", content.strip())
            if match:
                return round(float(match.group(0)), 2)
        except Exception:
            pass
        return 0.0
