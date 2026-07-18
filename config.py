"""
Central configuration, logging, and shared LLM utilities for ARIA.

This module is the single place where environment variables are loaded
(``load_dotenv`` runs exactly once, at import time) and where the retry
policy around LLM calls lives, so the individual nodes stay small.
"""

import logging
import os
import time

from dotenv import find_dotenv, load_dotenv

# Load environment variables exactly once, when this module is first imported.
# Every other module imports from here, so ``load_dotenv`` never runs per-node.
load_dotenv(find_dotenv())

# --- Constants -------------------------------------------------------------

GROQ_MODEL = "llama-3.1-8b-instant"

# The critic's ``overall`` score below which a finding is considered weak.
REPLAN_THRESHOLD = 7

# Hard cap on how many targeted replans a single run may perform.
MAX_REPLANS = 3

# LLM retry policy.
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


# --- Logging ---------------------------------------------------------------

_LOGGING_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, idempotently."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, ensuring logging is configured."""
    setup_logging()
    return logging.getLogger(name)


logger = get_logger("aria")


# --- Environment helpers ---------------------------------------------------


def require_groq_key() -> str:
    """
    Return the Groq API key or raise a clear error.

    Returns:
        str: The Groq API key from the environment.

    Raises:
        ValueError: If ``GROQ_API_KEY`` is not set.
    """
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "GROQ_API_KEY not found in environment. Set it in your .env file "
            "(see .env.example)."
        )
    return key


# --- LLM call helper -------------------------------------------------------


def invoke_with_retry(
    runnable,
    messages,
    *,
    context: str,
    max_retries: int = MAX_RETRIES,
    retry_delay: int = RETRY_DELAY_SECONDS,
):
    """
    Invoke a LangChain runnable with bounded retries.

    Works for both plain chat models (returns a message with ``.content``)
    and structured-output runnables (returns a Pydantic model). The caller
    is responsible for interpreting the returned object.

    Args:
        runnable: Any object exposing ``.invoke(messages)`` (a chat model or
            a ``with_structured_output`` runnable).
        messages: The messages / prompt to pass to ``invoke``.
        context (str): Human-readable label used in log messages.
        max_retries (int): Maximum attempts before giving up.
        retry_delay (int): Seconds to wait between attempts.

    Returns:
        The raw result of ``runnable.invoke(messages)`` on success, or
        ``None`` if every attempt failed.
    """
    for attempt in range(max_retries):
        try:
            return runnable.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - we deliberately retry on anything
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error(
                    "LLM call failed after %d retries in %s: %s",
                    max_retries,
                    context,
                    exc,
                )
                return None
