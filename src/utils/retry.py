"""
Retry utilities for external API calls.

Provides exponential backoff retry logic for unreliable network operations.

Usage:
    from src.utils.retry import retry_with_backoff, RetryConfig

    @retry_with_backoff(max_retries=3)
    def fetch_data():
        return requests.get(url)

    # Or with custom config
    config = RetryConfig(max_retries=5, base_delay=2.0)
    result = retry_with_backoff(fetch_data, config=config)
"""

import time
import random
import functools
from dataclasses import dataclass, field
from typing import Callable, TypeVar, Optional, Tuple, Type, Any
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    retryable_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (
            ConnectionError,
            TimeoutError,
            OSError,
        )
    )
    # HTTP status codes to retry
    retryable_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504)


# Default config for external API calls
DEFAULT_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
)


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate delay for exponential backoff.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    delay = min(
        config.base_delay * (config.exponential_base ** attempt),
        config.max_delay,
    )

    if config.jitter:
        # Add up to 25% random jitter
        delay = delay * (1 + random.uniform(0, 0.25))

    return delay


def should_retry_exception(exc: Exception, config: RetryConfig) -> bool:
    """Check if exception is retryable."""
    # Check if it's a requests exception with status code
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        return exc.response.status_code in config.retryable_status_codes

    # Check exception type
    return isinstance(exc, config.retryable_exceptions)


def retry_with_backoff(
    func: Optional[Callable[..., T]] = None,
    *,
    config: Optional[RetryConfig] = None,
    max_retries: Optional[int] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[..., T]:
    """
    Decorator/function for retrying with exponential backoff.

    Can be used as a decorator or called directly:

        # As decorator
        @retry_with_backoff(max_retries=3)
        def my_func():
            ...

        # Direct call
        result = retry_with_backoff(my_func, max_retries=3)

    Args:
        func: Function to retry
        config: Full retry configuration
        max_retries: Override for max retries (convenience)
        on_retry: Callback called on each retry (exc, attempt)

    Returns:
        Decorated function or result if called directly
    """
    if config is None:
        config = RetryConfig()
    if max_retries is not None:
        config.max_retries = max_retries

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc

                    if not should_retry_exception(exc, config):
                        logger.debug(f"Non-retryable exception: {type(exc).__name__}")
                        raise

                    if attempt < config.max_retries:
                        delay = calculate_delay(attempt, config)
                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_retries} for {fn.__name__} "
                            f"after {delay:.1f}s (error: {exc})"
                        )

                        if on_retry:
                            on_retry(exc, attempt)

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_retries} retries failed for {fn.__name__}"
                        )

            # Should never reach here, but for type safety
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic error")

        return wrapper

    # Handle both @retry_with_backoff and @retry_with_backoff()
    if func is not None:
        return decorator(func)
    return decorator


class RetryableRequest:
    """
    Context manager for retryable HTTP requests.

    Usage:
        with RetryableRequest(config) as session:
            response = session.get(url)

    Or manually:
        request = RetryableRequest()
        response = request.get(url)
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or DEFAULT_RETRY_CONFIG
        self._session = None

    def __enter__(self):
        import requests

        self._session = requests.Session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            self._session.close()
        return False

    def _make_request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> Any:
        """Make HTTP request with retry logic."""
        import requests

        session = self._session or requests

        # Set default timeout
        kwargs.setdefault("timeout", 30)

        @retry_with_backoff(config=self.config)
        def _request():
            response = getattr(session, method)(url, **kwargs)
            # Raise for retryable status codes
            if response.status_code in self.config.retryable_status_codes:
                response.raise_for_status()
            return response

        return _request()

    def get(self, url: str, **kwargs: Any) -> Any:
        """GET request with retry."""
        return self._make_request("get", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> Any:
        """POST request with retry."""
        return self._make_request("post", url, **kwargs)
