#!/usr/bin/env python3
"""
Unit tests for BusFactorCalculator functionality.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the project root to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.metrics.busfactor_calculator import BusFactorCalculator
from src.metrics.base import ModelContext


class TestBusFactorCalculator(unittest.TestCase):
    """Test BusFactorCalculator functionality."""
    
    def setUp(self):
        """Set up test calculator."""
        self.calculator = BusFactorCalculator()
    
    def test_calculator_initialization(self):
        """Test calculator is properly initialized."""
        self.assertEqual(self.calculator.name, "BusFactor")
        self.assertIsNone(self.calculator.get_score())
        self.assertIsNone(self.calculator.get_calculation_time())
    
    def test_extract_github_repo_info_valid(self):
        """Test GitHub repo info extraction with valid URLs."""
        test_cases = [
            ("https://github.com/owner/repo", {"owner": "owner", "repo": "repo"}),
            ("https://github.com/microsoft/DialoGPT", {"owner": "microsoft", "repo": "DialoGPT"}),
            ("https://github.com/pytorch/pytorch/", {"owner": "pytorch", "repo": "pytorch"}),
            ("https://github.com/owner/repo.git", {"owner": "owner", "repo": "repo"}),
        ]
        
        for url, expected in test_cases:
            with self.subTest(url=url):
                result = self.calculator._extract_github_repo_info(url)
                self.assertEqual(result, expected)
    
    def test_extract_github_repo_info_invalid(self):
        """Test GitHub repo info extraction with invalid URLs."""
        invalid_urls = [
            "https://huggingface.co/microsoft/DialoGPT-medium",
            "https://example.com/repo",
            "not-a-url",
            "",
            None
        ]
        
        for url in invalid_urls:
            with self.subTest(url=url):
                result = self.calculator._extract_github_repo_info(url) if url else None
                if url is None:
                    self.assertIsNone(result)
                else:
                    self.assertIsNone(result)
    
    @patch('src.metrics.busfactor_calculator.requests.get')
    def test_fetch_github_commits_success(self, mock_get):
        """Test successful GitHub commits fetching."""
        # Mock response data
        mock_commits = [
            {
                "author": {"login": "user1"},
                "commit": {"author": {"email": "user1@example.com"}}
            },
            {
                "author": {"login": "user2"},
                "commit": {"author": {"email": "user2@example.com"}}
            }
        ]
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_commits
        mock_get.return_value = mock_response
        
        result = self.calculator._fetch_github_commits_last_12_months("owner", "repo")
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["author"]["login"], "user1")
        self.assertEqual(result[1]["author"]["login"], "user2")
    
    @patch('src.metrics.busfactor_calculator.requests.get')
    def test_fetch_github_commits_api_error(self, mock_get):
        """Test GitHub commits fetching with API error."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_get.return_value = mock_response
        
        result = self.calculator._fetch_github_commits_last_12_months("owner", "repo")
        
        self.assertEqual(result, [])
    
    @patch('src.metrics.busfactor_calculator.BusFactorCalculator._get_contributors_last_12_months')
    def test_calculate_score_various_contributor_counts(self, mock_get_contributors):
        """Test score calculation with various contributor counts."""
        test_cases = [
            (0, 0.0),    # No contributors
            (5, 0.5),    # 5 contributors -> 5/10 = 0.5
            (10, 1.0),   # 10 contributors -> 10/10 = 1.0 (capped at 1.0)
            (25, 1.0),   # 25 contributors -> 25/10 = 2.5, but capped at 1.0
            (100, 1.0),  # 100 contributors -> still capped at 1.0
        ]
        
        context = ModelContext(
            model_url="https://github.com/owner/repo",
            model_info={"source": "github"},
            code_url="https://github.com/owner/repo"
        )
        
        for contributors, expected_score in test_cases:
            with self.subTest(contributors=contributors):
                mock_get_contributors.return_value = contributors
                score = self.calculator.calculate_score(context)
                self.assertEqual(score, expected_score)
    
    def test_calculate_score_no_code_url(self):
        """Test score calculation when no code URL is provided."""
        context = ModelContext(
            model_url="https://example.com",
            model_info={}
        )
        
        score = self.calculator.calculate_score(context)
        self.assertEqual(score, 0.0)
    
    def test_calculate_score_empty_context(self):
        """Test score calculation with None context."""
        score = self.calculator.calculate_score(None)
        self.assertEqual(score, 0.0)
    
    @patch('src.metrics.busfactor_calculator.BusFactorCalculator._fetch_github_commits_last_12_months')
    def test_get_contributors_unique_count(self, mock_fetch_commits):
        """Test that contributors are counted uniquely."""
        # Mock commits with duplicate contributors
        mock_commits = [
            {"author": {"login": "user1"}},
            {"author": {"login": "user2"}},
            {"author": {"login": "user1"}},  # Duplicate
            {"author": None, "commit": {"author": {"email": "user3@example.com"}}},
            {"author": None, "commit": {"author": {"email": "user3@example.com"}}},  # Duplicate email
        ]
        mock_fetch_commits.return_value = mock_commits
        
        count = self.calculator._get_contributors_last_12_months("https://github.com/owner/repo")
        
        # Should count user1, user2, and user3@example.com = 3 unique contributors
        self.assertEqual(count, 3)
    
    @patch('src.metrics.busfactor_calculator.BusFactorCalculator._extract_github_repo_info')
    def test_get_contributors_invalid_repo(self, mock_extract_repo):
        """Test contributors extraction with invalid repo info."""
        mock_extract_repo.return_value = None
        
        count = self.calculator._get_contributors_last_12_months("https://invalid-url.com")
        
        self.assertEqual(count, 0)


class TestBusFactorCalculatorIntegration(unittest.TestCase):
    """Integration tests for BusFactorCalculator with real-like scenarios."""
    
    def setUp(self):
        self.calculator = BusFactorCalculator()
    
    def test_github_repo_context(self):
        """Test with GitHub repository context."""
        context = ModelContext(
            model_url="https://github.com/microsoft/DialoGPT",
            model_info={"source": "github", "type": "repository"},
            code_url="https://github.com/microsoft/DialoGPT"
        )
        
        score = self.calculator.calculate_score(context)
        
        # Score should be between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Should have calculation time
        self.assertIsNotNone(self.calculator.get_calculation_time())
        self.assertGreater(self.calculator.get_calculation_time(), 0)
    
    def test_huggingface_model_without_code_url(self):
        """Test with HuggingFace model context (no code URL)."""
        context = ModelContext(
            model_url="https://huggingface.co/microsoft/DialoGPT-medium",
            model_info={"source": "huggingface", "type": "model"}
        )
        
        score = self.calculator.calculate_score(context)
        
        # Should return 0.0 since there's no code URL
        self.assertEqual(score, 0.0)


if __name__ == '__main__':
    unittest.main()