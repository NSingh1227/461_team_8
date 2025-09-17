#!/usr/bin/env python3
"""
Unit tests for URL processor functionality.
"""

import sys
import os
import unittest
from typing import List, Dict, Any

# Add the project root to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.url_processor import (
    URLProcessor, URLType, URLHandler, ModelHandler, DatasetHandler, CodeHandler,
    process_url, categorize_url, is_valid_url, fetch_huggingface_metadata, fetch_github_metadata
)
from src.metrics.base import ModelContext


class TestURLValidation(unittest.TestCase):
    """Test URL validation functionality."""
    
    def test_valid_urls(self):
        """Test that valid URLs are correctly identified."""
        valid_urls = [
            "https://huggingface.co/microsoft/DialoGPT-medium",
            "http://github.com/microsoft/DialoGPT",
            "https://example.com:8080/path",
            "https://example.com#section",
            "https://example.com?q=test#section"
        ]
        
        for url in valid_urls:
            with self.subTest(url=url):
                self.assertTrue(is_valid_url(url), f"URL should be valid: {url}")
    
    def test_invalid_urls(self):
        """Test that invalid URLs are correctly identified."""
        invalid_urls = [
            "huggingface.co/microsoft/DialoGPT-medium",  # No scheme
            "",  # Empty
            "not-a-url",  # Just text
            "https://",  # No domain
            "https://example .com",  # Space in URL
            "https:///invalid",  # Malformed
            "ftp://files.example.com",  # FTP protocol
            "file:///path/to/file"  # File protocol
        ]
        
        for url in invalid_urls:
            with self.subTest(url=url):
                self.assertFalse(is_valid_url(url), f"URL should be invalid: {url}")


class TestURLCategorization(unittest.TestCase):
    """Test URL categorization functionality."""
    
    def test_huggingface_model_urls(self):
        """Test HuggingFace model URL categorization."""
        model_urls = [
            "https://huggingface.co/microsoft/DialoGPT-medium",
            "https://huggingface.co/gpt2",
            "https://huggingface.co/microsoft/DialoGPT-medium/tree/main"
        ]
        
        for url in model_urls:
            with self.subTest(url=url):
                result = categorize_url(url)
                self.assertEqual(result, URLType.HUGGINGFACE_MODEL)
    
    def test_huggingface_dataset_urls(self):
        """Test HuggingFace dataset URL categorization."""
        dataset_urls = [
            "https://huggingface.co/datasets/squad",
            "https://huggingface.co/datasets/wikipedia",
            "https://huggingface.co/datasets/glue/cola"
        ]
        
        for url in dataset_urls:
            with self.subTest(url=url):
                result = categorize_url(url)
                self.assertEqual(result, URLType.HUGGINGFACE_DATASET)
    
    def test_github_urls(self):
        """Test GitHub repository URL categorization."""
        github_urls = [
            "https://github.com/microsoft/DialoGPT",
            "https://github.com/huggingface/transformers",
            "https://github.com/microsoft/DialoGPT/issues"
        ]
        
        for url in github_urls:
            with self.subTest(url=url):
                result = categorize_url(url)
                self.assertEqual(result, URLType.GITHUB_REPO)
    
    def test_unknown_urls(self):
        """Test unknown URL categorization."""
        unknown_urls = [
            "https://example.com/some/path",
            "https://gitlab.com/test/repo",
            "not-a-url"
        ]
        
        for url in unknown_urls:
            with self.subTest(url=url):
                result = process_url(url)
                self.assertEqual(result, URLType.UNKNOWN)


class TestURLHandlers(unittest.TestCase):
    """Test URL handler functionality."""
    
    def test_model_handler(self):
        """Test ModelHandler processing."""
        handler = ModelHandler()
        url = "https://huggingface.co/microsoft/DialoGPT-medium"
        
        context = handler.process_url(url)
        
        self.assertIsInstance(context, ModelContext)
        self.assertEqual(context.model_url, url)
        self.assertEqual(context.model_info["source"], "huggingface")
        self.assertEqual(context.model_info["type"], "model")
    
    def test_dataset_handler(self):
        """Test DatasetHandler processing."""
        handler = DatasetHandler()
        url = "https://huggingface.co/datasets/squad"
        
        context = handler.process_url(url)
        
        self.assertIsInstance(context, ModelContext)
        self.assertEqual(context.model_url, url)
        self.assertEqual(context.dataset_url, url)
        self.assertEqual(context.model_info["source"], "huggingface")
        self.assertEqual(context.model_info["type"], "dataset")
    
    def test_code_handler(self):
        """Test CodeHandler processing."""
        handler = CodeHandler()
        url = "https://github.com/microsoft/DialoGPT"
        
        context = handler.process_url(url)
        
        self.assertIsInstance(context, ModelContext)
        self.assertEqual(context.model_url, url)
        self.assertEqual(context.code_url, url)
        self.assertEqual(context.model_info["source"], "github")
        self.assertEqual(context.model_info["type"], "repository")


class TestURLProcessor(unittest.TestCase):
    """Test URLProcessor functionality."""
    
    def setUp(self):
        """Set up test files."""
        self.test_file = "test_urls.txt"
        self.test_urls = [
            "https://huggingface.co/microsoft/DialoGPT-medium",
            "https://huggingface.co/datasets/squad",
            "https://github.com/microsoft/DialoGPT"
        ]
        
        with open(self.test_file, 'w') as f:
            for url in self.test_urls:
                f.write(url + '\n')
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_read_urls(self):
        """Test URL reading from file."""
        processor = URLProcessor(self.test_file)
        urls = processor.read_urls()
        
        self.assertEqual(len(urls), len(self.test_urls))
        self.assertEqual(urls, self.test_urls)
    
    def test_process_urls(self):
        """Test URL processing."""
        processor = URLProcessor(self.test_file)
        results = processor.process_urls()
        
        self.assertEqual(len(results), len(self.test_urls))
        
        for result in results:
            self.assertIn('url', result)
            self.assertIn('type', result)
            self.assertIn(result['url'], self.test_urls)
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent files."""
        processor = URLProcessor("nonexistent_file.txt")
        urls = processor.read_urls()
        
        self.assertEqual(urls, [])


if __name__ == '__main__':
    unittest.main()