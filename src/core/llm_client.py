import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from .http_client import post_with_rate_limit
from .rate_limiter import APIService


def ask_for_json_score(prompt: str,
                       api_url: str = "https://genai.rcac.purdue.edu/api/chat/completions",
                       model: str = "llama3.1:latest") -> Tuple[Optional[float],
                                                                Optional[str]]:
    api_key: Optional[str] = os.getenv("GEN_AI_STUDIO_API_KEY")
    if not api_key:
        is_autograder: bool = os.environ.get('AUTOGRADER', '').lower() in [
            'true', '1', 'yes']
        debug_enabled: bool = os.environ.get('DEBUG', '').lower() in [
            'true', '1', 'yes']

        if not is_autograder and debug_enabled:
            print("[LLMClient] Missing GEN_AI_STUDIO_API_KEY. "
                  "Please set it in your environment.", file=sys.stderr)
        return None, "API key not available"

    try:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are an AI model evaluator. "
                                          "Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ]

        payload: Dict[str, Any] = {"model": model, "messages": messages}
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        response = post_with_rate_limit(
            api_url,
            APIService.GENAI,
            json=payload,
            headers=headers,
            timeout=30
        )

        if response and response.status_code == 200:
            data: Dict[str, Any] = response.json()
            content: str = data["choices"][0]["message"]["content"]
            return _extract_json_score(content)
        else:
            if response:
                is_autograder = os.environ.get('AUTOGRADER', '').lower() in [
                    'true', '1', 'yes']
                debug_enabled = os.environ.get('DEBUG', '').lower() in [
                    'true', '1', 'yes']

                if not is_autograder and debug_enabled:
                    print(f"[LLMClient] API error {response.status_code}: "
                          f"{response.text}", file=sys.stderr)
            return None, "API request failed"

    except Exception as e:
        is_autograder = os.environ.get('AUTOGRADER', '').lower() in ['true', '1', 'yes']
        debug_enabled = os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']

        if not is_autograder and debug_enabled:
            print(f"[LLMClient] Request failed: {e}", file=sys.stderr)
        return None, f"Request error: {str(e)}"


def _extract_json_score(content: str) -> Tuple[Optional[float], Optional[str]]:
    if not content:
        return None, "Empty response"

    try:
        data: Dict[str, Any] = json.loads(content.strip())
        score: Any = data.get("score")
        rationale: str = str(data.get("rationale", ""))

        if isinstance(score, (int, float)):
            score = float(score)
            score = max(0.0, min(1.0, score))
            return score, rationale

    except json.JSONDecodeError:
        pass

    json_match: Optional[re.Match[str]] = re.search(
        r'\{[^}]*"score"[^}]*\}', content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            score = data.get("score")
            rationale = str(data.get("rationale", ""))

            if isinstance(score, (int, float)):
                score = float(score)
                score = max(0.0, min(1.0, score))
                return score, rationale
        except json.JSONDecodeError:
            pass

    score_match: Optional[re.Match[str]] = re.search(
        r'Score:\s*(\d+(?:\.\d+)?)', content, re.IGNORECASE)
    if score_match:
        try:
            score_value: float = float(score_match.group(1))
            score_value = max(0.0, min(1.0, score_value))
            return score_value, content.strip()
        except ValueError:
            pass

    score_match = re.search(r'\b(\d+(?:\.\d+)?)\b', content)
    if score_match:
        try:
            score_value = float(score_match.group(1))
            score_value = max(0.0, min(1.0, score_value))
            return score_value, content.strip()
        except ValueError:
            pass

    return None, content.strip() if content else "No valid response"
