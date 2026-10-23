"""Root-level pytest configuration.

Ensures compatibility between AsyncMock and asyncio.iscoroutinefunction()
across Python versions. In Python 3.11, AsyncMock sets _is_coroutine on
instances but not on the class itself. Test helpers that access
AsyncMock._is_coroutine as a class attribute fail without this shim.
"""
import asyncio.coroutines
import os
from unittest.mock import AsyncMock

os.environ.setdefault("SR2_NO_SCOPE", "1")

# Expose the sentinel as a class attribute so that test helpers can do:
#   bridge._is_coroutine = AsyncMock._is_coroutine
AsyncMock._is_coroutine = asyncio.coroutines._is_coroutine
