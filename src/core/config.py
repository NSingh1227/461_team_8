import os
from typing import Optional

class Config:
    """Configuration management for API tokens and settings."""

    @staticmethod
    def get_github_token() -> Optional[str]:
        """Get GitHub token from environment or file."""
        token = os.environ.get('GITHUB_TOKEN')
        if token:
            return token
        try:
            with open('.github_token', 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            return None

    @staticmethod
    def get_genai_token() -> Optional[str]:
        """Get Purdue GenAI Studio API token from environment or file."""
        # Use the correct environment variable name as specified in the instructions
        token = os.environ.get('GEN_AI_STUDIO_API_KEY')
        if token:
            return token
        try:
            with open('.genai_token', 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            return None