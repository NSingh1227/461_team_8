#!/usr/bin/env python3
"""
Git repository analysis module using dulwich library.
Provides functionality to clone repositories and analyze Git metadata.
"""

import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from dulwich import porcelain
from dulwich.objects import Commit
from dulwich.repo import Repo


class GitAnalyzer:
    """Analyzer for Git repository metadata using dulwich library."""
    
    def __init__(self) -> None:
        """Initialize the Git analyzer."""
        self.temp_dirs: List[str] = []
    
    def clone_repository(self, repo_url: str, timeout: int = 30) -> Optional[str]:
        """
        Clone a Git repository to a temporary directory.
        
        Args:
            repo_url: URL of the Git repository to clone
            timeout: Timeout in seconds for the clone operation
            
        Returns:
            Path to the cloned repository or None if cloning failed
        """
        try:
            # Create temporary directory
            temp_dir: str = tempfile.mkdtemp(prefix="git_analysis_")
            self.temp_dirs.append(temp_dir)
            
            # Parse URL to get repository name
            parsed_url = urlparse(repo_url)
            repo_name: str = parsed_url.path.strip('/').split('/')[-1]
            if repo_name.endswith('.git'):
                repo_name = repo_name[:-4]
            
            repo_path: str = os.path.join(temp_dir, repo_name)
            
            # Clone repository
            start_time: float = time.time()
            porcelain.clone(repo_url, repo_path, depth=1)
            clone_time: float = time.time() - start_time
            
            if clone_time > timeout:
                print(f"Warning: Clone operation took {clone_time:.2f}s, exceeding timeout", file=sys.stderr)
                return None
                
            return repo_path
            
        except Exception as e:
            print(f"Error cloning repository {repo_url}: {e}", file=sys.stderr)
            return None
    
    def analyze_repository(self, repo_path: str) -> Dict[str, Any]:
        """
        Analyze a Git repository for metadata.
        
        Args:
            repo_path: Path to the Git repository
            
        Returns:
            Dictionary containing repository metadata
        """
        try:
            repo: Repo = Repo(repo_path)
            
            # Get basic repository information
            metadata: Dict[str, Any] = {
                "path": repo_path,
                "is_git_repo": True,
                "branch_count": 0,
                "commit_count": 0,
                "contributor_count": 0,
                "last_commit_date": None,
                "first_commit_date": None,
                "file_count": 0,
                "total_lines": 0,
                "languages": {},
                "has_readme": False,
                "has_license": False,
                "has_tests": False
            }
            
            # Count branches
            try:
                branches: List[bytes] = list(repo.get_refs())
                metadata["branch_count"] = len(branches)
            except Exception:
                pass
            
            # Analyze commits
            try:
                commits: List[Commit] = list(repo.get_walker())
                metadata["commit_count"] = len(commits)
                
                if commits:
                    # Get last commit date
                    last_commit: Commit = commits[0]
                    metadata["last_commit_date"] = last_commit.commit_time
                    
                    # Get first commit date
                    first_commit: Commit = commits[-1]
                    metadata["first_commit_date"] = first_commit.commit_time
                    
                    # Count unique contributors
                    contributors: set = set()
                    for commit in commits:
                        if commit.author:
                            contributors.add(commit.author)
                        if commit.committer:
                            contributors.add(commit.committer)
                    metadata["contributor_count"] = len(contributors)
                    
            except Exception:
                pass
            
            # Analyze files
            try:
                tree = repo[repo.head()]
                file_count: int = 0
                total_lines: int = 0
                languages: Dict[str, int] = {}
                
                for item in tree.items():
                    if item[1].type == 2:  # File
                        file_count += 1
                        
                        # Get file extension for language detection
                        filename: str = item[0].decode('utf-8')
                        if '.' in filename:
                            ext: str = filename.split('.')[-1].lower()
                            languages[ext] = languages.get(ext, 0) + 1
                        
                        # Check for specific files
                        if filename.lower() in ['readme.md', 'readme.txt', 'readme.rst']:
                            metadata["has_readme"] = True
                        elif filename.lower() in ['license', 'license.txt', 'license.md']:
                            metadata["has_license"] = True
                        elif 'test' in filename.lower() or filename.endswith('_test.py'):
                            metadata["has_tests"] = True
                
                metadata["file_count"] = file_count
                metadata["total_lines"] = total_lines
                metadata["languages"] = languages
                
            except Exception:
                pass
            
            return metadata
            
        except Exception as e:
            print(f"Error analyzing repository {repo_path}: {e}", file=sys.stderr)
            return {
                "path": repo_path,
                "is_git_repo": False,
                "error": str(e)
            }
    
    def analyze_github_repo(self, repo_url: str) -> Dict[str, Any]:
        """
        Analyze a GitHub repository by cloning it locally.
        
        Args:
            repo_url: GitHub repository URL
            
        Returns:
            Dictionary containing repository analysis results
        """
        # Clone repository
        repo_path: Optional[str] = self.clone_repository(repo_url)
        if not repo_path:
            return {
                "url": repo_url,
                "success": False,
                "error": "Failed to clone repository"
            }
        
        # Analyze repository
        metadata: Dict[str, Any] = self.analyze_repository(repo_path)
        metadata["url"] = repo_url
        metadata["success"] = True
        
        return metadata
    
    def cleanup(self) -> None:
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up {temp_dir}: {e}", file=sys.stderr)
        self.temp_dirs.clear()
    
    def __del__(self) -> None:
        """Cleanup on destruction."""
        self.cleanup()


def analyze_git_repository(repo_url: str) -> Dict[str, Any]:
    """
    Convenience function to analyze a Git repository.
    
    Args:
        repo_url: URL of the Git repository to analyze
        
    Returns:
        Dictionary containing repository analysis results
    """
    analyzer: GitAnalyzer = GitAnalyzer()
    try:
        return analyzer.analyze_github_repo(repo_url)
    finally:
        analyzer.cleanup()
