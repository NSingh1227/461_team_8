import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class APIService(Enum):
    """Enumeration of supported API services."""
    GITHUB = "github"
    GITLAB = "gitlab"
    HUGGINGFACE = "huggingface"
    GENAI = "genai"
    GENERAL_HTTP = "general"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting a specific API service."""
    requests_per_window: int
    window_seconds: int
    max_backoff_seconds: int = 300
    base_delay_seconds: float = 1.0


class RateLimiter:
    """Rate limiter for managing API request quotas across different services."""

    DEFAULT_CONFIGS: Dict[APIService, RateLimitConfig] = {
        APIService.GITHUB: RateLimitConfig(
            requests_per_window=60,
            window_seconds=3600,
            max_backoff_seconds=300,
            base_delay_seconds=1.0
        ),
        APIService.GITLAB: RateLimitConfig(
            requests_per_window=60,
            window_seconds=3600,
            max_backoff_seconds=300,
            base_delay_seconds=1.0
        ),
        APIService.HUGGINGFACE: RateLimitConfig(
            requests_per_window=100,
            window_seconds=3600,
            max_backoff_seconds=180,
            base_delay_seconds=0.5
        ),
        APIService.GENAI: RateLimitConfig(
            requests_per_window=30,
            window_seconds=60,
            max_backoff_seconds=600,
            base_delay_seconds=2.0
        ),
        APIService.GENERAL_HTTP: RateLimitConfig(
            requests_per_window=200,
            window_seconds=60,
            max_backoff_seconds=30,
            base_delay_seconds=0.1
        )
    }

    def __init__(self, custom_configs: Optional[Dict[APIService, RateLimitConfig]] = None) -> None:
        """Initialize rate limiter with default or custom configurations."""
        self._configs: Dict[APIService, RateLimitConfig] = self.DEFAULT_CONFIGS.copy()
        if custom_configs:
            self._configs.update(custom_configs)

        self._request_windows: Dict[APIService, deque] = {
            service: deque() for service in APIService
        }

        self._failure_counts: Dict[APIService, int] = {
            service: 0 for service in APIService
        }

        self._locks: Dict[APIService, threading.Lock] = {
            service: threading.Lock() for service in APIService
        }

    def check_quota(self, service: APIService) -> bool:
        """Check if quota is available for the given service."""
        with self._locks[service]:
            self._cleanup_old_requests(service)
            config: RateLimitConfig = self._configs[service]
            current_count: int = len(self._request_windows[service])
            return current_count < config.requests_per_window

    def has_quota(self, service: APIService) -> bool:
        """Check if quota is available for the given service."""
        return self.check_quota(service)

    def wait_if_needed(self, service: APIService) -> None:
        """Wait if necessary to respect rate limits before making a request."""
        with self._locks[service]:
            self._cleanup_old_requests(service)
            config: RateLimitConfig = self._configs[service]

            if len(self._request_windows[service]) >= config.requests_per_window:
                oldest_request: float = self._request_windows[service][0]
                wait_time: float = config.window_seconds - (time.time() - oldest_request)

                if wait_time > 0:
                    print(f"[RateLimiter] {service.value} quota exceeded. "
                          f"Waiting {wait_time:.1f}s", file=sys.stderr)
                    time.sleep(wait_time)
                    self._cleanup_old_requests(service)

            self._request_windows[service].append(time.time())

    def handle_rate_limit_response(self, service: APIService,
                                  retry_after: Optional[int] = None) -> None:
        """Handle rate limit response by implementing backoff strategy."""
        with self._locks[service]:
            self._failure_counts[service] += 1
            config: RateLimitConfig = self._configs[service]

            if retry_after:
                wait_time: float = min(retry_after, config.max_backoff_seconds)
                print(f"[RateLimiter] {service.value} rate limited. "
                      f"Server requested {retry_after}s wait, using {wait_time}s",
                      file=sys.stderr)
            else:
                backoff: float = (config.base_delay_seconds *
                                  (2 ** (self._failure_counts[service] - 1)))
                jitter: float = random.uniform(0, backoff * 0.1)
                wait_time = min(backoff + jitter, config.max_backoff_seconds)
                print(f"[RateLimiter] {service.value} rate limited. "
                      f"Exponential backoff: {wait_time:.1f}s "
                      f"(attempt {self._failure_counts[service]})", file=sys.stderr)

            time.sleep(wait_time)

    def reset_failures(self, service: APIService) -> None:
        """Reset failure count for a service after successful request."""
        with self._locks[service]:
            self._failure_counts[service] = 0

    def get_quota_status(self, service: APIService) -> Dict[str, Any]:
        """Get current quota status for a service."""
        with self._locks[service]:
            self._cleanup_old_requests(service)
            config: RateLimitConfig = self._configs[service]
            current_count: int = len(self._request_windows[service])

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
        """Remove old requests outside the time window."""
        config: RateLimitConfig = self._configs[service]
        current_time: float = time.time()
        window: deque = self._request_windows[service]

        while window and (current_time - window[0]) > config.window_seconds:
            window.popleft()


_rate_limiter_instance: Optional[RateLimiter] = None
_instance_lock: threading.Lock = threading.Lock()


def get_rate_limiter(config: Optional[Dict[APIService, RateLimitConfig]] = None) -> RateLimiter:
    """Get singleton rate limiter instance."""
    global _rate_limiter_instance

    if _rate_limiter_instance is None:
        with _instance_lock:
            if _rate_limiter_instance is None:
                _rate_limiter_instance = RateLimiter(config)

    return _rate_limiter_instance


def set_rate_limiter(rate_limiter: RateLimiter) -> None:
    """Set the global rate limiter instance (for testing)."""
    global _rate_limiter_instance
    with _instance_lock:
        _rate_limiter_instance = rate_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter instance (for testing)."""
    global _rate_limiter_instance
    with _instance_lock:
        _rate_limiter_instance = None
