import time
from functools import wraps
import openai
import logging

logger = logging.getLogger("trialmatcher")


def retry_with_exponential_backoff(max_retries=5, base_wait=1):
    """
    A decorator to apply exponential backoff for functions that might hit rate limits.

    :param max_retries: Maximum number of retries
    :param base_wait: Initial wait time (seconds) before retrying
    :return: Decorated function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt <= max_retries:
                try:
                    return func(*args, **kwargs)  # Call the wrapped function
                except openai.RateLimitError as e:
                    attempt += 1
                    if attempt > max_retries:
                        print("Maximum retry attempts reached. Exiting.")
                        raise e  # Re-raise the exception if retries are exhausted
                    wait_time = base_wait * (2 ** (attempt - 1))  # Exponential backoff
                    logger.info(
                        f"Rate limit hit in attempt #{attempt}. Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)

        return wrapper

    return decorator
