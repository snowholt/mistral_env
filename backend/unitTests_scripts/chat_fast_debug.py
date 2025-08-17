"""Deprecated: chat_fast_debug

The chat_fast wrapper has been removed. Use ChatService.chat(..., thinking_mode=False)
for low-latency tests. This script is retained as a stub to avoid broken references.
"""

from __future__ import annotations

import sys
print("chat_fast_debug.py is deprecated. Use ChatService.chat with thinking_mode=False instead.")
sys.exit(0)
