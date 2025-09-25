from typing import Optional


class TrustworthyModelException(Exception):
    pass


class MetricCalculationException(TrustworthyModelException):
    def __init__(self, metric_name: str, message: str,
                 original_exception: Optional[Exception] = None) -> None:
        self.metric_name: str = metric_name
        self.original_exception: Optional[Exception] = original_exception
        super().__init__(f"Failed to calculate {metric_name} metric: {message}")


class APIRateLimitException(TrustworthyModelException):
    def __init__(self, api_name: str, retry_after: Optional[int] = None) -> None:
        self.api_name: str = api_name
        self.retry_after: Optional[int] = retry_after
        message: str = f"Rate limit exceeded for {api_name} API"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message)


class InvalidURLException(TrustworthyModelException):
    def __init__(self, url: str, reason: str) -> None:
        self.url: str = url
        self.reason: str = reason
        super().__init__(f"Invalid URL '{url}': {reason}")


class ConfigurationException(TrustworthyModelException):
    pass
