import os
import sys
import re
import json
from typing import Dict, Any, Optional, Tuple
from .http_client import post_with_rate_limit
from .rate_limiter import APIService


def ask_for_json_score(prompt: str, 
                      api_url: str = "https://genai.rcac.purdue.edu/api/chat/completions",
                      model: str = "llama3.1:latest") -> Tuple[Optional[float], Optional[str]]:
    """
    Ask the Purdue GenAI Studio API for a JSON response with score and rationale.
    
    Returns:
        Tuple of (score: float or None, rationale: str or None)
    """
    api_key = os.getenv("GEN_AI_STUDIO_API_KEY")
    if not api_key:
        # Check if we're in an autograder environment or if debug output is disabled
        is_autograder = os.environ.get('AUTOGRADER', '').lower() in ['true', '1', 'yes']
        debug_enabled = os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']
        
        if not is_autograder and debug_enabled:
            print("[LLMClient] Missing GEN_AI_STUDIO_API_KEY. Please set it in your environment.", file=sys.stderr)
        return None, "API key not available"
    
    try:
        messages = [
            {"role": "system", "content": "You are an AI model evaluator. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        payload = {"model": model, "messages": messages}
        headers = {
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
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return _extract_json_score(content)
        else:
            if response:
                # Check if we're in an autograder environment or if debug output is disabled
                is_autograder = os.environ.get('AUTOGRADER', '').lower() in ['true', '1', 'yes']
                debug_enabled = os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']
                
                if not is_autograder and debug_enabled:
                    print(f"[LLMClient] API error {response.status_code}: {response.text}", file=sys.stderr)
            return None, "API request failed"
            
    except Exception as e:
        # Check if we're in an autograder environment or if debug output is disabled
        is_autograder = os.environ.get('AUTOGRADER', '').lower() in ['true', '1', 'yes']
        debug_enabled = os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']
        
        if not is_autograder and debug_enabled:
            print(f"[LLMClient] Request failed: {e}", file=sys.stderr)
        return None, f"Request error: {str(e)}"


def _extract_json_score(content: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract score and rationale from LLM response.
    Expects JSON format: {"score": float, "rationale": string}
    """
    if not content:
        return None, "Empty response"
    
    try:
        # First try to parse as JSON directly
        data = json.loads(content.strip())
        score = data.get("score")
        rationale = data.get("rationale", "")
        
        if isinstance(score, (int, float)):
            return float(score), str(rationale)
        
    except json.JSONDecodeError:
        pass
    
    # Fallback: try to extract JSON from text
    json_match = re.search(r'\{[^}]*"score"[^}]*\}', content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            score = data.get("score")
            rationale = data.get("rationale", "")
            
            if isinstance(score, (int, float)):
                return float(score), str(rationale)
        except json.JSONDecodeError:
            pass
    
    # Final fallback: extract numeric score with regex
    # Look for "Score:" followed by a number
    score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', content, re.IGNORECASE)
    if score_match:
        try:
            score = float(score_match.group(1))
            return score, content.strip()
        except ValueError:
            pass
    
    # Alternative: look for any decimal number
    score_match = re.search(r'\b(\d+(?:\.\d+)?)\b', content)
    if score_match:
        try:
            score = float(score_match.group(1))
            return score, content.strip()
        except ValueError:
            pass
    
    return None, content.strip() if content else "No valid response"