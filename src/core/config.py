import os
from typing import Optional


class Config:
    """Configuration management for the trustworthy model analyzer."""

    @staticmethod
    def get_github_token() -> Optional[str]:
        """Get GitHub token from environment variable or file."""
        token: Optional[str] = os.environ.get('GITHUB_TOKEN')
        if token:
            return token
        try:
            with open('.github_token', 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            return None

    @staticmethod
    def get_genai_token() -> Optional[str]:
        """Get GenAI Studio API key from environment variable or file."""
        token: Optional[str] = os.environ.get('GEN_AI_STUDIO_API_KEY')
        if token:
            return token
        try:
            with open('.genai_token', 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            return None
