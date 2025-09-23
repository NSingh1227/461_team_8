#!/usr/bin/env python3
"""
Unit tests for rate limiter functionality.
"""

import sys
import os
import unittest
import time
from unittest.mock import patch, MagicMock, Mock
import threading

# Add the project root to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.rate_limiter import RateLimiter, APIService, RateLimitConfig, get_rate_limiter, reset_rate_limiter, set_rate_limiter
from src.core.http_client import get_with_rate_limit, make_rate_limited_request


class TestRateLimiter(unittest.TestCase):
    """Test RateLimiter functionality."""
    
    def setUp(self):
        """Set up test rate limiter with fast settings for testing."""
        self.test_config = {
            APIService.GITHUB: RateLimitConfig(
                requests_per_window=3,
                window_seconds=2,
                max_backoff_seconds=5,
                base_delay_seconds=0.1
            )
        }
        self.rate_limiter = RateLimiter(self.test_config)
    
    def test_initialization(self):
        """Test rate limiter is properly initialized."""
        self.assertIsInstance(self.rate_limiter, RateLimiter)
        self.assertTrue(self.rate_limiter.check_quota(APIService.GITHUB))
    
    def test_quota_tracking(self):
        """Test quota tracking works correctly."""
        service = APIService.GITHUB
        
        # Should be within quota initially
        self.assertTrue(self.rate_limiter.check_quota(service))
        
        # Make requests up to the limit
        for i in range(3):
            self.rate_limiter.wait_if_needed(service)
            self.assertEqual(
                self.rate_limiter.get_quota_status(service)['current_requests'], 
                i + 1
            )
        
        # Should now be at quota limit
        self.assertFalse(self.rate_limiter.check_quota(service))
    
    def test_sliding_window(self):
        """Test sliding window cleanup works."""
        service = APIService.GITHUB
        
        # Fill quota
        for _ in range(3):
            self.rate_limiter.wait_if_needed(service)
        
        self.assertFalse(self.rate_limiter.check_quota(service))
        
        # Wait for window to expire
        time.sleep(2.1)
        
        # Should be within quota again
        self.assertTrue(self.rate_limiter.check_quota(service))
    
    def test_quota_status(self):
        """Test quota status reporting."""
        service = APIService.GITHUB
        status = self.rate_limiter.get_quota_status(service)
        
        self.assertEqual(status['service'], 'github')
        self.assertEqual(status['current_requests'], 0)
        self.assertEqual(status['max_requests'], 3)
        self.assertEqual(status['quota_remaining'], 3)
        self.assertTrue(status['within_quota'])
        
        # Make a request
        self.rate_limiter.wait_if_needed(service)
        status = self.rate_limiter.get_quota_status(service)
        
        self.assertEqual(status['current_requests'], 1)
        self.assertEqual(status['quota_remaining'], 2)
    
    def test_failure_tracking(self):
        """Test failure count tracking and reset."""
        service = APIService.GITHUB
        
        # Simulate failures
        self.rate_limiter.handle_rate_limit_response(service)
        status = self.rate_limiter.get_quota_status(service)
        self.assertEqual(status['failure_count'], 1)
        
        self.rate_limiter.handle_rate_limit_response(service)
        status = self.rate_limiter.get_quota_status(service)
        self.assertEqual(status['failure_count'], 2)
        
        # Reset failures
        self.rate_limiter.reset_failures(service)
        status = self.rate_limiter.get_quota_status(service)
        self.assertEqual(status['failure_count'], 0)
    
    def test_thread_safety(self):
        """Test rate limiter is thread-safe."""
        service = APIService.GITHUB
        results = []
        
        def worker():
            try:
                self.rate_limiter.wait_if_needed(service)
                results.append(True)
            except Exception as e:
                results.append(False)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have completed without errors
        self.assertEqual(len(results), 5)
        self.assertTrue(all(results))


class TestRateLimiterSingleton(unittest.TestCase):
    """Test rate limiter singleton functionality."""
    
    def setUp(self):
        """Reset singleton before each test."""
        reset_rate_limiter()
    
    def tearDown(self):
        """Reset singleton after each test."""
        reset_rate_limiter()
    
    def test_singleton_pattern(self):
        """Test singleton pattern works correctly."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        
        self.assertIs(limiter1, limiter2)
    
    def test_reset_singleton(self):
        """Test singleton reset works."""
        limiter1 = get_rate_limiter()
        reset_rate_limiter()
        limiter2 = get_rate_limiter()
        
        self.assertIsNot(limiter1, limiter2)


class TestHTTPClient(unittest.TestCase):
    """Test HTTP client with rate limiting."""
    
    def setUp(self):
        """Set up test environment."""
        reset_rate_limiter()
    
    def tearDown(self):
        """Clean up after tests."""
        reset_rate_limiter()
    
    @patch('src.core.http_client.requests.request')
    def test_successful_request(self, mock_request):
        """Test successful HTTP request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_request.return_value = mock_response
        
        response = get_with_rate_limit("https://api.github.com/test", APIService.GITHUB)
        
        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 200)
        mock_request.assert_called_once()
    
    @patch('src.core.http_client.requests.request')
    @patch('time.sleep')
    def test_rate_limit_retry(self, mock_sleep, mock_request):
        """Test rate limit handling with retry."""
        # First call returns 429, second returns 200
        mock_responses = []
        
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {'Retry-After': '1'}
        mock_responses.append(rate_limit_response)
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {"success": True}
        mock_responses.append(success_response)
        
        mock_request.side_effect = mock_responses
        
        response = get_with_rate_limit("https://api.github.com/test", APIService.GITHUB)
        
        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_request.call_count, 2)
        mock_sleep.assert_called()
    
    @patch('src.core.http_client.requests.request')
    def test_server_error_retry(self, mock_request):
        """Test server error retry logic."""
        # Mock server error followed by success
        mock_responses = []
        
        error_response = Mock()
        error_response.status_code = 500
        mock_responses.append(error_response)
        
        success_response = Mock()
        success_response.status_code = 200
        mock_responses.append(success_response)
        
        mock_request.side_effect = mock_responses
        
        response = get_with_rate_limit("https://api.github.com/test", APIService.GITHUB)
        
        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_request.call_count, 2)
    
    @patch('src.core.http_client.requests.request')
    def test_client_error_no_retry(self, mock_request):
        """Test client errors are not retried."""
        # Mock 404 error
        mock_response = Mock()
        mock_response.status_code = 404
        mock_request.return_value = mock_response
        
        response = get_with_rate_limit("https://api.github.com/test", APIService.GITHUB)
        
        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 404)
        mock_request.assert_called_once()


class TestRateLimiterIntegration(unittest.TestCase):
    """Integration tests for rate limiter with realistic scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        reset_rate_limiter()
        # Use very small limits for fast testing
        self.test_config = {
            APIService.GITHUB: RateLimitConfig(
                requests_per_window=2,
                window_seconds=1,
                max_backoff_seconds=2,
                base_delay_seconds=0.1
            )
        }
    
    def tearDown(self):
        """Clean up after tests."""
        reset_rate_limiter()
    
    @patch('src.core.http_client.requests.request')
    def test_quota_enforcement(self, mock_request):
        """Test that quota is properly enforced."""
        # Set up rate limiter with test config
        rate_limiter = RateLimiter(self.test_config)
        set_rate_limiter(rate_limiter)
        
        # Mock successful responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response
        
        start_time = time.time()
        
        # Make requests beyond quota (2 requests allowed per 1 second window)
        # First 2 requests should be immediate, 3rd and 4th should wait
        for i in range(4):
            response = get_with_rate_limit("https://api.github.com/test", APIService.GITHUB)
            self.assertIsNotNone(response)
        
        elapsed = time.time() - start_time
        
        # Should have taken at least 0.9 seconds due to rate limiting
        # (waiting for the window to reset after 2 requests)
        self.assertGreater(elapsed, 0.9)
    
    def test_different_services_independent(self):
        """Test that different services have independent quotas."""
        rate_limiter = RateLimiter(self.test_config)
        
        # Fill GitHub quota
        for _ in range(2):
            rate_limiter.wait_if_needed(APIService.GITHUB)
        
        # HuggingFace should still have quota
        self.assertTrue(rate_limiter.check_quota(APIService.HUGGINGFACE))
        self.assertFalse(rate_limiter.check_quota(APIService.GITHUB))


if __name__ == '__main__':
    unittest.main()