import os
import re
import requests
from typing import Dict, Any, Optional


class LLMAnalyzer:
    """
    Analyzer that uses Purdue GenAI Studio API (OpenAI-compatible)
    to evaluate model metadata, README files, and datasets.
    """

    def __init__(self,
                 api_url: str = "https://genai.rcac.purdue.edu/api/chat/completions",
                 model: str = "llama3.1:latest",
                 api_key: Optional[str] = None):
        self.api_url = api_url
        self.model = model
        self.api_key = api_key or os.getenv("GENAI_API_KEY")

    def _post_to_genai(self, messages: list[Dict[str, str]]) -> Optional[str]:
        if not self.api_key:
            print("[LLMAnalyzer] Missing GENAI_API_KEY. Please set it in your environment.")
            return None
        try:
            payload = {"model": self.model, "messages": messages}
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                print(f"[LLMAnalyzer] API error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"[LLMAnalyzer] Request failed: {e}")
            return None

    def analyze_dataset_quality(self, dataset_info: Dict[str, Any]) -> float:
        user_prompt = (
            "Analyze the following dataset information and return ONLY "
            "a numeric quality score between 0.0 and 1.0:\n\n"
            f"{dataset_info}"
        )
        messages = [
            {"role": "system", "content": "You are an evaluator that scores dataset quality."},
            {"role": "user", "content": user_prompt}
        ]
        content = self._post_to_genai(messages)
        return self._extract_score(content)

    def _extract_score(self, content: Optional[str]) -> float:
        if not content:
            return 0.0
        try:
            # Strict float extraction
            match = re.search(r"\b(0(\.\d+)?|1(\.0+)?)\b", content.strip())
            if match:
                return round(float(match.group(1)), 2)
        except Exception:
            pass
        return 0.0
