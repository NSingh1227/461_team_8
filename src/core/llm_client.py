import os
import json
from typing import Optional, Tuple
import requests


GENAI_API_URL = "https://genai.rcac.purdue.edu/api/chat/completions"
DEFAULT_MODEL = "llama3.1:latest"


def call_llm(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 30) -> Optional[str]:
    """
    Call the GenAI OpenAI-compatible endpoint with a user prompt.
    Returns the assistant message content as a string, or None on error.
    """
    api_key = os.environ.get("GENAI_API_KEY")
    if not api_key:
        print("[LLM] Warning: API key not set (RCAC_GENAI_API_KEY). Skipping LLM call.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise evaluator. Always return concise JSON when asked."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    try:
        resp = requests.post(GENAI_API_URL, headers=headers, json=body, timeout=timeout)
        if resp.status_code != 200:
            print(f"[LLM] Error: {resp.status_code} {resp.text[:200]}")
            return None
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        content = message.get("content")
        return content
    except Exception as e:
        print(f"[LLM] Exception: {e}")
        return None


def ask_for_json_score(prompt: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Ask the LLM to return a JSON object: {"score": number in [0,1], "rationale": string}.
    Returns (score, rationale). If parsing fails, returns (None, None).
    """
    wrapped_prompt = (
        "Return a JSON object with keys 'score' (float 0..1) and 'rationale' (string).\n\n"
        + prompt
    )
    content = call_llm(wrapped_prompt)
    if not content:
        return None, None
    try:
        # Try to locate a JSON object in the response
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = content[start : end + 1]
            obj = json.loads(json_str)
            score = obj.get("score")
            rationale = obj.get("rationale")
            if isinstance(score, (int, float)):
                score_f = max(0.0, min(1.0, float(score)))
                return score_f, rationale if isinstance(rationale, str) else None
    except Exception as e:
        print(f"[LLM] Parse error: {e}")
    return None, None


