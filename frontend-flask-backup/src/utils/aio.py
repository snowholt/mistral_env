import asyncio
from typing import Any, Awaitable


def run_async(coro: Awaitable[Any]):
    """Run an async coroutine in a fresh event loop (Flask sync context)."""
    return asyncio.run(coro)
