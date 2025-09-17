#!/usr/bin/env python3
"""
Unit tests for license calculation functionality.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the project root to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.metrics.license_calculator import LicenseCalculator
from src.metrics.base import ModelContext


class TestLicenseCalculator(unittest.TestCase):
    """Test LicenseCalculator functionality."""
    
    def setUp(self):
        """Set up test calculator."""
        self.calculator = LicenseCalculator()
    
    def test_calculator_initialization(self):
        """Test calculator is properly initialized."""
        self.assertEqual(self.calculator.name, "License")
        self.assertIsNone(self.calculator.get_score())
        self.assertIsNone(self.calculator.get_calculation_time())
    
    def test_lgpl_compatibility_scores(self):
        """Test LGPL compatibility score mappings."""
        test_cases = [
            ("mit", 1.0),
            ("apache-2.0", 1.0),
            ("bsd-3-clause", 1.0),
            ("lgpl-2.1", 1.0),
            ("gpl-3.0", 0.0),
            ("agpl-3.0", 0.0),
            ("proprietary", 0.0),
            ("unknown-license", 0.5)
        ]
        
        for license_text, expected_score in test_cases:
            with self.subTest(license=license_text):
                score = self.calculator._calculate_compatibility_score(license_text)
                self.assertEqual(score, expected_score, 
                               f"License {license_text} should score {expected_score}")
    
    def test_calculate_compatibility_score_none(self):
        """Test compatibility score calculation with None input."""
        score = self.calculator._calculate_compatibility_score(None)
        self.assertEqual(score, 0.5)
    
    def test_calculate_compatibility_score_empty(self):
        """Test compatibility score calculation with empty string."""
        score = self.calculator._calculate_compatibility_score("")
        self.assertEqual(score, 0.5)
    
    def test_calculate_compatibility_score_case_insensitive(self):
        """Test that license matching is case insensitive."""
        test_cases = [
            ("MIT", 1.0),
            ("Mit", 1.0),
            ("APACHE-2.0", 1.0),
            ("GPL-3.0", 0.0),
            ("Gpl-3.0", 0.0)
        ]
        
        for license_text, expected_score in test_cases:
            with self.subTest(license=license_text):
                score = self.calculator._calculate_compatibility_score(license_text)
                self.assertEqual(score, expected_score)
    
    def test_calculate_compatibility_score_partial_match(self):
        """Test partial license name matching."""
        test_cases = [
            ("MIT License", 1.0),
            ("Apache License 2.0", 1.0),
            ("GPL-3.0 License", 0.0),  # Contains gpl-3.0 as substring
            ("Custom MIT-based license", 1.0)
        ]
        
        for license_text, expected_score in test_cases:
            with self.subTest(license=license_text):
                score = self.calculator._calculate_compatibility_score(license_text)
                self.assertEqual(score, expected_score)
    
    def test_extract_repo_id_valid(self):
        """Test repository ID extraction from valid URLs."""
        test_cases = [
            ("https://huggingface.co/microsoft/DialoGPT-medium", "microsoft/DialoGPT-medium"),
            ("https://huggingface.co/gpt2", "gpt2"),
            ("https://huggingface.co/datasets/squad", "datasets/squad")
        ]
        
        for url, expected_repo_id in test_cases:
            with self.subTest(url=url):
                repo_id = self.calculator._extract_repo_id(url)
                self.assertEqual(repo_id, expected_repo_id)
    
    def test_extract_repo_id_invalid(self):
        """Test repository ID extraction from invalid URLs."""
        invalid_urls = [
            "https://github.com/microsoft/DialoGPT",
            "https://example.com/test",
            "not-a-url"
        ]
        
        for url in invalid_urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    self.calculator._extract_repo_id(url)
    
    def test_extract_license_from_readme(self):
        """Test license extraction from README content."""
        test_cases = [
            ("License: MIT\nThis is a test project", "mit"),
            ("license: apache-2.0\nSome content", "apache-2.0"),
            ("No license info here", None),
            ("", None)
        ]
        
        for readme_content, expected_license in test_cases:
            with self.subTest(content=readme_content[:20]):
                license_text = self.calculator._extract_license_from_readme(readme_content)
                self.assertEqual(license_text, expected_license)
    
    def test_extract_huggingface_license_with_metadata(self):
        """Test license extraction from HuggingFace metadata."""
        # Test with cardData license
        context_with_card = ModelContext(
            model_url="https://huggingface.co/test/model",
            model_info={},
            huggingface_metadata={
                "cardData": {"license": "MIT"}
            }
        )
        
        license_text = self.calculator._extract_huggingface_license(context_with_card)
        self.assertEqual(license_text, "mit")
        
        # Test with tags license
        context_with_tags = ModelContext(
            model_url="https://huggingface.co/test/model",
            model_info={},
            huggingface_metadata={
                "tags": ["license:apache-2.0", "other-tag"]
            }
        )
        
        license_text = self.calculator._extract_huggingface_license(context_with_tags)
        self.assertEqual(license_text, "apache-2.0")
    
    def test_extract_github_license(self):
        """Test license extraction from GitHub metadata."""
        context_with_github = ModelContext(
            model_url="https://github.com/test/repo",
            model_info={
                "github_metadata": {
                    "license": {
                        "spdx_id": "MIT",
                        "name": "MIT License"
                    }
                }
            }
        )
        
        license_text = self.calculator._extract_github_license(context_with_github)
        self.assertEqual(license_text, "mit")
        
        # Test with name only
        context_name_only = ModelContext(
            model_url="https://github.com/test/repo",
            model_info={
                "github_metadata": {
                    "license": {
                        "name": "Apache License 2.0"
                    }
                }
            }
        )
        
        license_text = self.calculator._extract_github_license(context_name_only)
        self.assertEqual(license_text, "apache license 2.0")
    
    @patch('src.metrics.license_calculator.time.time')
    def test_calculate_score_timing(self, mock_time):
        """Test that calculation timing is properly recorded."""
        # Mock time to return specific values
        mock_time.side_effect = [0.0, 0.1]  # Start and end times
        
        context = ModelContext(
            model_url="https://example.com/test",
            model_info={},
            huggingface_metadata={"cardData": {"license": "MIT"}}
        )
        
        score = self.calculator.calculate_score(context)
        
        # Should have calculated a score and timing
        self.assertIsNotNone(self.calculator.get_score())
        self.assertEqual(self.calculator.get_calculation_time(), 100)  # 0.1 seconds = 100ms
    
    def test_calculate_score_with_exception(self):
        """Test score calculation handles exceptions gracefully."""
        # Create a context that will cause an exception
        context = ModelContext(
            model_url="invalid-url",
            model_info=None,
            huggingface_metadata=None
        )
        
        score = self.calculator.calculate_score(context)
        
        # Should return default score of 0.5 when exceptions occur
        self.assertEqual(score, 0.5)
        self.assertIsNotNone(self.calculator.get_calculation_time())


class TestLicenseCalculatorIntegration(unittest.TestCase):
    """Integration tests for LicenseCalculator with real-like data."""
    
    def setUp(self):
        """Set up test calculator."""
        self.calculator = LicenseCalculator()
    
    def test_huggingface_model_context(self):
        """Test license calculation for HuggingFace model context."""
        context = ModelContext(
            model_url="https://huggingface.co/microsoft/DialoGPT-medium",
            model_info={"source": "huggingface", "type": "model"},
            huggingface_metadata={
                "cardData": {"license": "MIT"}
            }
        )
        
        score = self.calculator.calculate_score(context)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsNotNone(self.calculator.get_calculation_time())
    
    def test_github_repo_context(self):
        """Test license calculation for GitHub repository context."""
        context = ModelContext(
            model_url="https://github.com/microsoft/DialoGPT",
            model_info={
                "source": "github",
                "type": "repository",
                "github_metadata": {
                    "license": {
                        "spdx_id": "MIT",
                        "name": "MIT License"
                    }
                }
            },
            code_url="https://github.com/microsoft/DialoGPT"
        )
        
        score = self.calculator.calculate_score(context)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsNotNone(self.calculator.get_calculation_time())
    
    def test_unknown_license_context(self):
        """Test license calculation for context without license info."""
        context = ModelContext(
            model_url="https://example.com/unknown",
            model_info={"source": "unknown"},
            huggingface_metadata=None
        )
        
        score = self.calculator.calculate_score(context)
        
        # Should return default score
        self.assertEqual(score, 0.5)
        self.assertIsNotNone(self.calculator.get_calculation_time())


if __name__ == '__main__':
    unittest.main()