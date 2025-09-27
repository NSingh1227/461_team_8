import os
import sys
from typing import Any, Optional

import requests

from .rate_limiter import APIService, get_rate_limiter

_session: requests.Session = requests.Session()
_session.headers.update({
    'User-Agent': 'ECE461-Team8-ModelAnalyzer/1.0'
})

hf_token = os.environ.get('HF_API_TOKEN')
if hf_token:
    _session.headers.update({
        'Authorization': f'Bearer {hf_token}'
    })


def make_rate_limited_request(
    method: str,
    url: str,
    service: APIService,
    max_retries: int = 3,
    **kwargs: Any
) -> Optional[requests.Response]:
    rate_limiter = get_rate_limiter()

    for attempt in range(max_retries + 1):
        try:
            rate_limiter.wait_if_needed(service)

            response: requests.Response = _session.request(method, url, **kwargs)

            if response.status_code == 200:
                rate_limiter.reset_failures(service)
                return response

            elif response.status_code == 429:
                retry_after: Optional[int] = None
                if 'Retry-After' in response.headers:
                    try:
                        retry_after = int(response.headers['Retry-After'])
                    except ValueError:
                        pass

                rate_limiter.handle_rate_limit_response(service, retry_after)
                continue

            elif response.status_code in [500, 502, 503, 504]:
                if attempt < max_retries:
                    rate_limiter.handle_rate_limit_response(service)
                    continue
                else:
                    print(f"[HTTPClient] Server error {response.status_code} after "
                          f"{max_retries} retries: {url}", file=sys.stderr)
                    return response

            else:
                return response

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                print(f"[HTTPClient] Request exception on attempt {attempt + 1}: {e}",
                      file=sys.stderr)
                rate_limiter.handle_rate_limit_response(service)
                continue
            else:
                print(f"[HTTPClient] Request failed after {max_retries} retries: {e}",
                      file=sys.stderr)
                return None

    return None


def get_with_rate_limit(url: str, service: APIService,
                        **kwargs: Any) -> Optional[requests.Response]:
    return make_rate_limited_request('GET', url, service, **kwargs)


def post_with_rate_limit(url: str, service: APIService,
                         **kwargs: Any) -> Optional[requests.Response]:
    return make_rate_limited_request('POST', url, service, **kwargs)


def head_with_rate_limit(url: str, service: APIService,
                         **kwargs: Any) -> Optional[requests.Response]:
    return make_rate_limited_request('HEAD', url, service, **kwargs)
