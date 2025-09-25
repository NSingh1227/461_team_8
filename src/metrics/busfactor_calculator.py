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
    """Calculator for bus factor metric - measures knowledge concentration."""

    def __init__(self) -> None:
        super().__init__("BusFactor")

    def calculate_score(self, context: ModelContext) -> float:
        """Calculate bus factor score based on contributor diversity."""
        start_time: float = time.time()

        try:
            url_to_use: Optional[str] = context.code_url or context.model_url

            if not url_to_use or not url_to_use.startswith("https://github.com"):
                score: float = 0.0
            else:
                # Try GitHub API first
                contributors_count: int = self._get_contributors_last_12_months(url_to_use)

                # If GitHub API fails, try local Git analysis
                if contributors_count == 0:
                    contributors_count = self._get_contributors_from_local_git(url_to_use)

                if contributors_count <= 5:
                    score = contributors_count / 10.0
                else:
                    score = 0.5 + (contributors_count - 5) / 20.0
                score = min(1.0, score)

        except Exception as e:
            print(f"Error calculating Bus Factor score: {e}", file=sys.stderr)
            score = 0.0

        end_time: float = time.time()
        calculation_time_ms: int = int((end_time - start_time) * 1000)
        self._set_score(score, calculation_time_ms)

        return score

    def _get_contributors_from_local_git(self, code_url: str) -> int:
        """Get contributor count using local Git analysis as fallback."""
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
        """Get number of unique contributors in the last 12 months."""
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
                if commit.get('author') and commit['author'].get('login'):
                    contributors.add(commit['author']['login'])
                elif commit.get('commit', {}).get('author', {}).get('email'):
                    contributors.add(commit['commit']['author']['email'])

            return len(contributors)

        except Exception as e:
            print(f"Error getting contributors: {e}", file=sys.stderr)
            return 0

    def _get_historical_contributors(self, owner: str, repo: str) -> int:
        """Get historical contributors count as fallback."""
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
        """Extract owner and repo name from GitHub URL."""
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
        """Fetch commits from the last 12 months."""
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

            commits: List[Dict[str, Any]] = response.json()
            return commits[:50]

        except Exception as e:
            print(f"Error fetching GitHub commits: {e}", file=sys.stderr)
            return []

