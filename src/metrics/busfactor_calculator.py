import re, json, time, requests, sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from .base import MetricCalculator, ModelContext
from ..core.config import Config
from ..core.http_client import get_with_rate_limit
from ..core.rate_limiter import APIService

class BusFactorCalculator(MetricCalculator):
    
    def __init__(self):
        super().__init__("BusFactor")
    
    def calculate_score(self, context: ModelContext) -> float:
        start_time = time.time()
        
        try:

            url_to_use = context.code_url or context.model_url
            
            if not url_to_use or not url_to_use.startswith("https://github.com"):
                score = 0.0
            else:
                contributors_count = self._get_contributors_last_12_months(url_to_use)

                if contributors_count <= 5:
                    score = contributors_count / 10.0
                else:
                    score = 0.5 + (contributors_count - 5) / 20.0
                score = min(1.0, score)
                
        except Exception as e:
            print(f"Error calculating Bus Factor score: {e}", file=sys.stderr)
            score = 0.0
        
        end_time = time.time()
        calculation_time_ms = int((end_time - start_time) * 1000)
        self._set_score(score, calculation_time_ms)
        
        return score
    
    def _get_contributors_last_12_months(self, code_url: str) -> int:
        try:
            repo_info = self._extract_github_repo_info(code_url)
            if not repo_info:
                return 0
            
            commits = self._fetch_github_commits_last_12_months(repo_info['owner'], repo_info['repo'])
            if not commits:

                return self._get_historical_contributors(repo_info['owner'], repo_info['repo'])
            
            contributors = set()
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
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
            headers = {'Accept': 'application/vnd.github.v3+json'}
            github_token = Config.get_github_token()
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            
            params = {'per_page': 10, 'page': 1}
            
            response = get_with_rate_limit(url, APIService.GITHUB, headers=headers, params=params, timeout=10)
            
            if not response or response.status_code != 200:
                return 0
            
            contributors = response.json()


            return min(len(contributors), 8)
            
        except Exception as e:
            print(f"Error getting historical contributors: {e}", file=sys.stderr)
            return 0
    
    def _extract_github_repo_info(self, code_url: str) -> Optional[Dict[str, str]]:
        try:
            match = re.match(r'https?://github\.com/([^/]+)/([^/]+)/?', code_url)
            if match:
                owner, repo = match.groups()
                repo = repo.replace('.git', '')
                return {'owner': owner, 'repo': repo}
            return None
        except Exception:
            return None
    
    def _fetch_github_commits_last_12_months(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        try:
            twelve_months_ago = datetime.now() - timedelta(days=365)
            since_date = twelve_months_ago.isoformat()
            
            url = f"https://api.github.com/repos/{owner}/{repo}/commits"
            
            headers = {'Accept': 'application/vnd.github.v3+json'}
            github_token = Config.get_github_token()
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            
            params = {
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
            
            commits = response.json()
            return commits[:50]
            
        except Exception as e:
            print(f"Error fetching GitHub commits: {e}", file=sys.stderr)
            return []
        
