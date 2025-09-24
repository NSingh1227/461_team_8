import requests
import time
from typing import Optional, Dict, Any
from .rate_limiter import get_rate_limiter, APIService

# Create a session with connection pooling for better performance
_session = requests.Session()
_session.headers.update({
    'User-Agent': 'ECE461-Team8-ModelAnalyzer/1.0'
})


def make_rate_limited_request(
    method: str,
    url: str, 
    service: APIService,
    max_retries: int = 3,
    **kwargs
) -> Optional[requests.Response]:
    """
    Make an HTTP request with automatic rate limiting and retry logic.
    
    Args:
        method: HTTP method ('GET', 'POST', etc.)
        url: Request URL
        service: API service type for rate limiting
        max_retries: Maximum number of retry attempts
        **kwargs: Additional arguments passed to requests
        
    Returns:
        Response object or None if all retries failed
    """
    rate_limiter = get_rate_limiter()
    
    for attempt in range(max_retries + 1):
        try:
            # Wait if needed to respect rate limits
            rate_limiter.wait_if_needed(service)
            
            # Make the request using the session for connection pooling
            response = _session.request(method, url, **kwargs)
            
            # Handle different response codes
            if response.status_code == 200:
                # Success - reset failure count
                rate_limiter.reset_failures(service)
                return response
                
            elif response.status_code == 429:
                # Rate limited - handle with backoff
                retry_after = None
                if 'Retry-After' in response.headers:
                    try:
                        retry_after = int(response.headers['Retry-After'])
                    except ValueError:
                        pass
                
                rate_limiter.handle_rate_limit_response(service, retry_after)
                
                # Don't count rate limit as a retry attempt
                continue
                
            elif response.status_code in [500, 502, 503, 504]:
                # Server errors - retry with backoff
                if attempt < max_retries:
                    rate_limiter.handle_rate_limit_response(service)
                    continue
                else:
                    print(f"[HTTPClient] Server error {response.status_code} after {max_retries} retries: {url}")
                    return response
                    
            else:
                # Other errors - return immediately
                return response
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                print(f"[HTTPClient] Request exception on attempt {attempt + 1}: {e}")
                rate_limiter.handle_rate_limit_response(service)
                continue
            else:
                print(f"[HTTPClient] Request failed after {max_retries} retries: {e}")
                return None
    
    return None


def get_with_rate_limit(url: str, service: APIService, **kwargs) -> Optional[requests.Response]:
    """Convenience method for GET requests with rate limiting."""
    return make_rate_limited_request('GET', url, service, **kwargs)


def post_with_rate_limit(url: str, service: APIService, **kwargs) -> Optional[requests.Response]:
    """Convenience method for POST requests with rate limiting."""
    return make_rate_limited_request('POST', url, service, **kwargs)


def head_with_rate_limit(url: str, service: APIService, **kwargs) -> Optional[requests.Response]:
    """Convenience method for HEAD requests with rate limiting."""
    return make_rate_limited_request('HEAD', url, service, **kwargs)