import logging
import time
import typing as t
from contextlib import contextmanager


@contextmanager
def timer(logger: logging.Logger, action_name: str) -> t.Generator[None, None, None]:
    """A simple timer that logs the time take for the action"""

    logger.info(f"Starting {action_name}")
    start = time.perf_counter()

    yield

    duration = time.perf_counter() - start
    logger.info(f"{action_name} finished, duration {duration:.2f} seconds.")
