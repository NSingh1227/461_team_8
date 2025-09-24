import time
import threading
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import random


class APIService(Enum):
    """Supported API services with their default rate limits."""
    GITHUB = "github"
    HUGGINGFACE = "huggingface" 
    GENAI = "genai"
    GENERAL_HTTP = "general"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting per service."""
    requests_per_window: int
    window_seconds: int
    max_backoff_seconds: int = 300
    base_delay_seconds: float = 1.0
    

class RateLimiter:
    """Centralized rate limiter for API requests with sliding window and exponential backoff."""
    
    # Default rate limits based on typical API quotas
    DEFAULT_CONFIGS = {
        APIService.GITHUB: RateLimitConfig(
            requests_per_window=60,  # GitHub allows 60 req/hour for unauthenticated
            window_seconds=3600,     # 1 hour window
            max_backoff_seconds=300,
            base_delay_seconds=1.0
        ),
        APIService.HUGGINGFACE: RateLimitConfig(
            requests_per_window=100,  # Conservative limit
            window_seconds=3600,      # 1 hour window  
            max_backoff_seconds=180,
            base_delay_seconds=0.5
        ),
        APIService.GENAI: RateLimitConfig(
            requests_per_window=30,   # Conservative for GenAI
            window_seconds=60,        # 1 minute window
            max_backoff_seconds=600,
            base_delay_seconds=2.0
        ),
        APIService.GENERAL_HTTP: RateLimitConfig(
            requests_per_window=200,  # Fair for general HTTP
            window_seconds=60,        # 1 minute window
            max_backoff_seconds=30,
            base_delay_seconds=0.1
        )
    }
    
    def __init__(self, custom_configs: Optional[Dict[APIService, RateLimitConfig]] = None):
        """Initialize rate limiter with optional custom configurations."""
        self._configs = self.DEFAULT_CONFIGS.copy()
        if custom_configs:
            self._configs.update(custom_configs)
            
        # Sliding window storage: service -> deque of timestamps
        self._request_windows: Dict[APIService, deque] = {
            service: deque() for service in APIService
        }
        
        # Track consecutive failures for exponential backoff
        self._failure_counts: Dict[APIService, int] = {
            service: 0 for service in APIService
        }
        
        # Thread safety
        self._locks: Dict[APIService, threading.Lock] = {
            service: threading.Lock() for service in APIService
        }
        
    def check_quota(self, service: APIService) -> bool:
        """Check if we're within quota for the given service."""
        with self._locks[service]:
            self._cleanup_old_requests(service)
            config = self._configs[service]
            current_count = len(self._request_windows[service])
            return current_count < config.requests_per_window
    
    def has_quota(self, service: APIService) -> bool:
        """Alias for check_quota for backward compatibility."""
        return self.check_quota(service)
    
    def wait_if_needed(self, service: APIService) -> None:
        """Wait if necessary to respect rate limits for the given service."""
        with self._locks[service]:
            self._cleanup_old_requests(service)
            config = self._configs[service]
            
            # Check if we need to wait
            if len(self._request_windows[service]) >= config.requests_per_window:
                # Calculate wait time until oldest request expires
                oldest_request = self._request_windows[service][0]
                wait_time = config.window_seconds - (time.time() - oldest_request)
                
                if wait_time > 0:
                    print(f"[RateLimiter] {service.value} quota exceeded. Waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    # Clean up again after waiting
                    self._cleanup_old_requests(service)
            
            # Record this request
            self._request_windows[service].append(time.time())
    
    def handle_rate_limit_response(self, service: APIService, retry_after: Optional[int] = None) -> None:
        """Handle 429 Too Many Requests response with exponential backoff."""
        with self._locks[service]:
            self._failure_counts[service] += 1
            config = self._configs[service]
            
            # Use retry_after from response if provided, otherwise calculate backoff
            if retry_after:
                wait_time = min(retry_after, config.max_backoff_seconds)
                print(f"[RateLimiter] {service.value} rate limited. Server requested {retry_after}s wait, using {wait_time}s")
            else:
                # Exponential backoff: base_delay * 2^(failures-1) + jitter
                backoff = config.base_delay_seconds * (2 ** (self._failure_counts[service] - 1))
                jitter = random.uniform(0, backoff * 0.1)  # 10% jitter
                wait_time = min(backoff + jitter, config.max_backoff_seconds)
                print(f"[RateLimiter] {service.value} rate limited. Exponential backoff: {wait_time:.1f}s (attempt {self._failure_counts[service]})")
            
            time.sleep(wait_time)
    
    def reset_failures(self, service: APIService) -> None:
        """Reset failure count after successful request."""
        with self._locks[service]:
            self._failure_counts[service] = 0
    
    def get_quota_status(self, service: APIService) -> Dict[str, any]:
        """Get current quota status for debugging/monitoring."""
        with self._locks[service]:
            self._cleanup_old_requests(service)
            config = self._configs[service]
            current_count = len(self._request_windows[service])
            
            return {
                "service": service.value,
                "current_requests": current_count,
                "max_requests": config.requests_per_window,
                "window_seconds": config.window_seconds,
                "quota_remaining": config.requests_per_window - current_count,
                "failure_count": self._failure_counts[service],
                "within_quota": current_count < config.requests_per_window
            }
    
    def _cleanup_old_requests(self, service: APIService) -> None:
        """Remove requests outside the sliding window."""
        config = self._configs[service]
        current_time = time.time()
        window = self._request_windows[service]
        
        # Remove requests older than the window
        while window and (current_time - window[0]) > config.window_seconds:
            window.popleft()


# Global singleton instance
_rate_limiter_instance: Optional[RateLimiter] = None
_instance_lock = threading.Lock()


def get_rate_limiter(config: Optional[Dict[APIService, RateLimitConfig]] = None) -> RateLimiter:
    """Get the global rate limiter instance (singleton pattern)."""
    global _rate_limiter_instance
    
    if _rate_limiter_instance is None:
        with _instance_lock:
            if _rate_limiter_instance is None:
                _rate_limiter_instance = RateLimiter(config)
    
    return _rate_limiter_instance


def set_rate_limiter(rate_limiter: RateLimiter) -> None:
    """Set a custom rate limiter instance (mainly for testing)."""
    global _rate_limiter_instance
    with _instance_lock:
        _rate_limiter_instance = rate_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter instance (mainly for testing)."""
    global _rate_limiter_instance
    with _instance_lock:
        _rate_limiter_instance = None