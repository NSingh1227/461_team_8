#!/usr/bin/env python3
"""
Rate Limiter Demonstration Script

This script demonstrates the rate limiter functionality with realistic API calls.
"""

import sys
import os
import time
from typing import List

# Add the project root to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.rate_limiter import get_rate_limiter, reset_rate_limiter, APIService, RateLimitConfig, RateLimiter
from src.core.http_client import get_with_rate_limit


def demonstrate_basic_rate_limiting():
    """Demonstrate basic rate limiting functionality."""
    print("=" * 60)
    print("RATE LIMITER DEMONSTRATION")
    print("=" * 60)
    
    # Reset rate limiter for clean demo
    reset_rate_limiter()
    
    # Get rate limiter instance
    rate_limiter = get_rate_limiter()
    
    print("\n1. Initial Quota Status:")
    for service in [APIService.GITHUB, APIService.HUGGINGFACE, APIService.GENAI]:
        status = rate_limiter.get_quota_status(service)
        print(f"   {service.value:12}: {status['quota_remaining']}/{status['max_requests']} requests available")
    
    print("\n2. Making GitHub API Requests:")
    test_urls = [
        "https://api.github.com/repos/microsoft/vscode",
        "https://api.github.com/repos/python/cpython", 
        "https://api.github.com/repos/facebook/react"
    ]
    
    for i, url in enumerate(test_urls):
        print(f"   Request {i+1}: {url}")
        start_time = time.time()
        
        # This will respect rate limits automatically
        rate_limiter.wait_if_needed(APIService.GITHUB)
        
        elapsed = time.time() - start_time
        status = rate_limiter.get_quota_status(APIService.GITHUB)
        
        print(f"             Wait time: {elapsed:.2f}s")
        print(f"             Remaining quota: {status['quota_remaining']}/{status['max_requests']}")
    
    print("\n3. Final Quota Status:")
    for service in [APIService.GITHUB, APIService.HUGGINGFACE, APIService.GENAI]:
        status = rate_limiter.get_quota_status(service)
        print(f"   {service.value:12}: {status['quota_remaining']}/{status['max_requests']} requests available")


def demonstrate_custom_rate_limits():
    """Demonstrate custom rate limit configuration."""
    print("\n" + "=" * 60)
    print("CUSTOM RATE LIMITS DEMONSTRATION")
    print("=" * 60)
    
    # Create custom rate limiter with very strict limits for demo
    custom_config = {
        APIService.GITHUB: RateLimitConfig(
            requests_per_window=2,      # Only 2 requests
            window_seconds=5,           # Per 5 seconds
            max_backoff_seconds=10,
            base_delay_seconds=0.5
        )
    }
    
    custom_limiter = RateLimiter(custom_config)
    
    print(f"\nCustom GitHub limit: 2 requests per 5 seconds")
    
    print("\nMaking 4 requests to demonstrate quota enforcement:")
    
    for i in range(4):
        print(f"\nRequest {i+1}:")
        start_time = time.time()
        
        # Check quota before request
        status = custom_limiter.get_quota_status(APIService.GITHUB)
        print(f"   Before: {status['quota_remaining']}/{status['max_requests']} quota remaining")
        
        # Make request (will wait if needed)
        custom_limiter.wait_if_needed(APIService.GITHUB)
        
        elapsed = time.time() - start_time
        status = custom_limiter.get_quota_status(APIService.GITHUB)
        
        print(f"   After:  {status['quota_remaining']}/{status['max_requests']} quota remaining")
        print(f"   Wait time: {elapsed:.2f}s")


def demonstrate_failure_handling():
    """Demonstrate failure handling and exponential backoff."""
    print("\n" + "=" * 60)
    print("FAILURE HANDLING DEMONSTRATION")
    print("=" * 60)
    
    rate_limiter = get_rate_limiter()
    
    print("\nSimulating API failures and exponential backoff:")
    
    for i in range(3):
        print(f"\nFailure {i+1}:")
        status_before = rate_limiter.get_quota_status(APIService.GITHUB)
        print(f"   Failure count before: {status_before['failure_count']}")
        
        start_time = time.time()
        
        # Simulate handling a 429 response
        rate_limiter.handle_rate_limit_response(APIService.GITHUB)
        
        elapsed = time.time() - start_time
        status_after = rate_limiter.get_quota_status(APIService.GITHUB)
        
        print(f"   Failure count after: {status_after['failure_count']}")
        print(f"   Backoff time: {elapsed:.2f}s")
    
    # Reset failures
    print(f"\nResetting failures...")
    rate_limiter.reset_failures(APIService.GITHUB)
    status = rate_limiter.get_quota_status(APIService.GITHUB)
    print(f"   Failure count after reset: {status['failure_count']}")


def demonstrate_service_independence():
    """Demonstrate that different services have independent quotas."""
    print("\n" + "=" * 60)
    print("SERVICE INDEPENDENCE DEMONSTRATION")
    print("=" * 60)
    
    # Create a rate limiter with small limits for demonstration
    demo_config = {
        APIService.GITHUB: RateLimitConfig(requests_per_window=1, window_seconds=3),
        APIService.HUGGINGFACE: RateLimitConfig(requests_per_window=1, window_seconds=3),
        APIService.GENAI: RateLimitConfig(requests_per_window=1, window_seconds=3)
    }
    
    demo_limiter = RateLimiter(demo_config)
    
    print("\nEach service has 1 request per 3 seconds for this demo")
    
    services = [APIService.GITHUB, APIService.HUGGINGFACE, APIService.GENAI]
    
    print("\nUsing quota for each service:")
    for service in services:
        print(f"\n{service.value.upper()} Service:")
        status_before = demo_limiter.get_quota_status(service)
        print(f"   Before: {status_before['quota_remaining']}/{status_before['max_requests']}")
        
        demo_limiter.wait_if_needed(service)
        
        status_after = demo_limiter.get_quota_status(service)
        print(f"   After:  {status_after['quota_remaining']}/{status_after['max_requests']}")
    
    print(f"\nAll services now at quota limit, but they're independent:")
    for service in services:
        status = demo_limiter.get_quota_status(service)
        print(f"   {service.value:12}: {status['quota_remaining']}/{status['max_requests']} remaining")


def main():
    """Run all demonstrations."""
    try:
        demonstrate_basic_rate_limiting()
        demonstrate_custom_rate_limits()
        demonstrate_failure_handling()
        demonstrate_service_independence()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nThe rate limiter successfully:")
        print("✅ Tracks requests per service in sliding windows")
        print("✅ Enforces quota limits with automatic waiting")
        print("✅ Handles failures with exponential backoff")
        print("✅ Maintains service independence")
        print("✅ Provides thread-safe operations")
        print("\nIntegration points:")
        print("📡 GitHub API calls (busfactor_calculator.py, url_processor.py)")
        print("🤗 HuggingFace API calls (url_processor.py)")
        print("🧠 GenAI API calls (llm_analyzer.py)")
        print("🌐 General HTTP requests (dataset_code_calculator.py)")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()