class TrustworthyModelException(Exception):
    pass


class MetricCalculationException(TrustworthyModelException):
    def __init__(self, metric_name: str, message: str, original_exception: Exception = None):
        self.metric_name = metric_name
        self.original_exception = original_exception
        super().__init__(f"Failed to calculate {metric_name} metric: {message}")


class APIRateLimitException(TrustworthyModelException):
    def __init__(self, api_name: str, retry_after: int = None):
        self.api_name = api_name
        self.retry_after = retry_after
        message = f"Rate limit exceeded for {api_name} API"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message)


class InvalidURLException(TrustworthyModelException):
    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"Invalid URL '{url}': {reason}")


class ConfigurationException(TrustworthyModelException):
    pass
