import time
import sys
import threading
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import random


class APIService(Enum):
    GITHUB = "github"
    HUGGINGFACE = "huggingface" 
    GENAI = "genai"
    GENERAL_HTTP = "general"


@dataclass
class RateLimitConfig:
    requests_per_window: int
    window_seconds: int
    max_backoff_seconds: int = 300
    base_delay_seconds: float = 1.0
    

class RateLimiter:
    

    DEFAULT_CONFIGS = {
        APIService.GITHUB: RateLimitConfig(
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
    
    def __init__(self, custom_configs: Optional[Dict[APIService, RateLimitConfig]] = None):
        self._configs = self.DEFAULT_CONFIGS.copy()
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
        with self._locks[service]:
            self._cleanup_old_requests(service)
            config = self._configs[service]
            current_count = len(self._request_windows[service])
            return current_count < config.requests_per_window
    
    def has_quota(self, service: APIService) -> bool:
        return self.check_quota(service)
    
    def wait_if_needed(self, service: APIService) -> None:
        with self._locks[service]:
            self._cleanup_old_requests(service)
            config = self._configs[service]
            

            if len(self._request_windows[service]) >= config.requests_per_window:

                oldest_request = self._request_windows[service][0]
                wait_time = config.window_seconds - (time.time() - oldest_request)
                
                if wait_time > 0:
                    print(f"[RateLimiter] {service.value} quota exceeded. Waiting {wait_time:.1f}s", file=sys.stderr)
                    time.sleep(wait_time)

                    self._cleanup_old_requests(service)
            

            self._request_windows[service].append(time.time())
    
    def handle_rate_limit_response(self, service: APIService, retry_after: Optional[int] = None) -> None:
        with self._locks[service]:
            self._failure_counts[service] += 1
            config = self._configs[service]
            

            if retry_after:
                wait_time = min(retry_after, config.max_backoff_seconds)
                print(f"[RateLimiter] {service.value} rate limited. Server requested {retry_after}s wait, using {wait_time}s", file=sys.stderr)
            else:

                backoff = config.base_delay_seconds * (2 ** (self._failure_counts[service] - 1))
                jitter = random.uniform(0, backoff * 0.1)
                wait_time = min(backoff + jitter, config.max_backoff_seconds)
                print(f"[RateLimiter] {service.value} rate limited. Exponential backoff: {wait_time:.1f}s (attempt {self._failure_counts[service]})", file=sys.stderr)
            
            time.sleep(wait_time)
    
    def reset_failures(self, service: APIService) -> None:
        with self._locks[service]:
            self._failure_counts[service] = 0
    
    def get_quota_status(self, service: APIService) -> Dict[str, any]:
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
        config = self._configs[service]
        current_time = time.time()
        window = self._request_windows[service]
        

        while window and (current_time - window[0]) > config.window_seconds:
            window.popleft()



_rate_limiter_instance: Optional[RateLimiter] = None
_instance_lock = threading.Lock()


def get_rate_limiter(config: Optional[Dict[APIService, RateLimitConfig]] = None) -> RateLimiter:
    global _rate_limiter_instance
    
    if _rate_limiter_instance is None:
        with _instance_lock:
            if _rate_limiter_instance is None:
                _rate_limiter_instance = RateLimiter(config)
    
    return _rate_limiter_instance


def set_rate_limiter(rate_limiter: RateLimiter) -> None:
    global _rate_limiter_instance
    with _instance_lock:
        _rate_limiter_instance = rate_limiter


def reset_rate_limiter() -> None:
    global _rate_limiter_instance
    with _instance_lock:
        _rate_limiter_instance = None
