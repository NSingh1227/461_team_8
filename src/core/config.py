import os
from typing import Optional

class Config:

    @staticmethod
    def get_github_token() -> Optional[str]:
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
        token = os.environ.get('GEN_AI_STUDIO_API_KEY')
        if token:
            return token
        try:
            with open('.genai_token', 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            return None
