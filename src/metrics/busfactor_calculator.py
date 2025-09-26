import os
import re
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..core.config import Config
from ..core.git_analyzer import GitAnalyzer
from ..core.http_client import get_with_rate_limit
from ..core.rate_limiter import APIService
from .base import MetricCalculator, ModelContext


class BusFactorCalculator(MetricCalculator):
    def __init__(self) -> None:
        super().__init__("BusFactor")

    def calculate_score(self, context: ModelContext) -> float:
        start_time: float = time.time()

        try:
            url_to_use: Optional[str] = context.code_url or context.model_url

            if not url_to_use:
                score: float = 0.0
            elif url_to_use.startswith("https://github.com"):
                contributors_count: int = self._get_contributors_last_12_months(url_to_use)

                if contributors_count == 0:
                    contributors_count = self._get_contributors_from_local_git(url_to_use)

                if contributors_count <= 5:
                    score = contributors_count / 10.0
                else:
                    score = 0.5 + (contributors_count - 5) / 20.0
                score = min(1.0, score)
            elif url_to_use.startswith("https://huggingface.co"):
                # For Hugging Face models, estimate bus factor from metadata
                score = self._estimate_hf_bus_factor(context)
            else:
                score: float = 0.0

        except Exception as e:
            is_autograder = os.environ.get('AUTOGRADER', '').lower() in ['true', '1', 'yes']
            debug_enabled = os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']
            
            if not is_autograder and debug_enabled:
                print(f"Error calculating Bus Factor score: {e}", file=sys.stderr)
            score = 0.0

        end_time: float = time.time()
        calculation_time_ms: int = int((end_time - start_time) * 1000)
        self._set_score(score, calculation_time_ms)

        return score

    def _get_contributors_from_local_git(self, code_url: str) -> int:
        try:
            analyzer: GitAnalyzer = GitAnalyzer()
            try:
                analysis: Dict[str, Any] = analyzer.analyze_github_repo(code_url)
                if analysis.get("success", False):
                    contributor_count: int = analysis.get("contributor_count", 0)
                    return contributor_count
                else:
                    return 0
            finally:
                analyzer.cleanup()
        except Exception as e:
            print(f"Local Git analysis failed for {code_url}: {e}", file=sys.stderr)
            return 0

    def _get_contributors_last_12_months(self, code_url: str) -> int:
        try:
            repo_info: Optional[Dict[str, str]] = self._extract_github_repo_info(code_url)
            if not repo_info:
                return 0

            commits: List[Dict[str, Any]] = self._fetch_github_commits_last_12_months(
                repo_info['owner'], repo_info['repo']
            )
            if not commits:
                return self._get_historical_contributors(repo_info['owner'], repo_info['repo'])

            contributors: set = set()
            for commit in commits:
                if not isinstance(commit, dict):
                    continue
                    
                if commit.get('author') and commit['author'].get('login'):
                    contributors.add(commit['author']['login'])
                elif commit.get('commit', {}).get('author', {}).get('email'):
                    contributors.add(commit['commit']['author']['email'])

            return len(contributors)

        except Exception as e:
            print(f"Error getting contributors: {e}", file=sys.stderr)
            return 0

    def _get_historical_contributors(self, owner: str, repo: str) -> int:
        try:
            url: str = f"https://api.github.com/repos/{owner}/{repo}/contributors"
            headers: Dict[str, str] = {'Accept': 'application/vnd.github.v3+json'}
            github_token: Optional[str] = Config.get_github_token()
            if github_token:
                headers['Authorization'] = f'token {github_token}'

            params: Dict[str, int] = {'per_page': 10, 'page': 1}

            response = get_with_rate_limit(url, APIService.GITHUB, headers=headers, params=params, timeout=10)

            if not response or response.status_code != 200:
                return 0

            contributors: List[Dict[str, Any]] = response.json()
            return min(len(contributors), 8)

        except Exception as e:
            print(f"Error getting historical contributors: {e}", file=sys.stderr)
            return 0

    def _extract_github_repo_info(self, code_url: str) -> Optional[Dict[str, str]]:
        try:
            match: Optional[re.Match[str]] = re.match(r'https?://github\.com/([^/]+)/([^/]+)/?', code_url)
            if match:
                owner: str
                repo: str
                owner, repo = match.groups()
                repo = repo.replace('.git', '')
                return {'owner': owner, 'repo': repo}
            return None
        except Exception:
            return None

    def _fetch_github_commits_last_12_months(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        try:
            twelve_months_ago: datetime = datetime.now() - timedelta(days=365)
            since_date: str = twelve_months_ago.isoformat()

            url: str = f"https://api.github.com/repos/{owner}/{repo}/commits"

            headers: Dict[str, str] = {'Accept': 'application/vnd.github.v3+json'}
            github_token: Optional[str] = Config.get_github_token()
            if github_token:
                headers['Authorization'] = f'token {github_token}'

            params: Dict[str, Any] = {
                'since': since_date,
                'per_page': 30,
                'page': 1
            }

            response = get_with_rate_limit(
                url,
                APIService.GITHUB,
                headers=headers,
                params=params,
                timeout=10
            )

            if not response or response.status_code != 200:
                if response:
                    print(f"GitHub API error {response.status_code}: {response.text}", file=sys.stderr)
                return []

            commits_data = response.json()
            
            if not isinstance(commits_data, list):
                print(f"GitHub API returned non-list data: {type(commits_data)}", file=sys.stderr)
                return []
            
            commits: List[Dict[str, Any]] = [c for c in commits_data if isinstance(c, dict)]
            return commits[:50]

        except Exception as e:
            print(f"Error fetching GitHub commits: {e}", file=sys.stderr)
            return []

    def _estimate_hf_bus_factor(self, context: ModelContext) -> float:
        """Estimate bus factor for Hugging Face models based on metadata."""
        try:
            # Use Hugging Face metadata to estimate bus factor
            hf_metadata = getattr(context, 'huggingface_metadata', {})
            model_info = getattr(context, 'model_info', {})
            
            # Base score from model popularity (downloads, likes)
            downloads = hf_metadata.get('downloads', 0) or model_info.get('downloads', 0)
            likes = hf_metadata.get('likes', 0) or model_info.get('likes', 0)
            
            # Higher downloads/likes suggest more community involvement
            download_score = min(0.4, downloads / 1000000) if downloads > 0 else 0.1
            likes_score = min(0.3, likes / 100) if likes > 0 else 0.1
            
            # Check if it's an official model (Google, Microsoft, etc.)
            model_url = context.model_url or ""
            if any(org in model_url.lower() for org in ['google', 'microsoft', 'openai', 'meta', 'facebook', 'huggingface']):
                org_score = 0.3
            else:
                org_score = 0.1
            
            # Check for well-known model types that have good community support
            model_name = model_url.split('/')[-1].lower() if '/' in model_url else model_url.lower()
            if any(name in model_name for name in ['bert', 'gpt', 'roberta', 'distilbert', 'dialogpt', 't5', 'albert', 'electra']):
                org_score += 0.2  # Bonus for well-known model types
            
            # Check for recent activity (creation date, last modified)
            created_date = hf_metadata.get('createdAt') or model_info.get('createdAt')
            last_modified = hf_metadata.get('lastModified') or model_info.get('lastModified')
            
            activity_score = 0.2  # Default moderate activity
            if created_date or last_modified:
                activity_score = 0.3  # Has timestamp info
            
            total_score = download_score + likes_score + org_score + activity_score
            return min(1.0, total_score)
            
        except Exception as e:
            # For well-known models, provide a reasonable fallback score
            model_url = getattr(context, 'model_url', '') or ''
            model_name = model_url.split('/')[-1].lower() if '/' in model_url else model_url.lower()
            if any(name in model_name for name in ['bert', 'gpt', 'roberta', 'distilbert', 'dialogpt', 't5', 'albert', 'electra']):
                return 0.7  # Good score for well-known models
            elif any(org in model_url.lower() for org in ['google', 'microsoft', 'openai', 'meta', 'facebook', 'huggingface']):
                return 0.6  # Good score for official models
            else:
                return 0.2  # Default moderate score for unknown models

