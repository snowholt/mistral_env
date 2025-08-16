import os
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """Load application configuration from environment variables.

    Returns a mapping merged with sensible defaults.
    """
    return {
        "SECRET_KEY": os.environ.get("WEBUI_SECRET_KEY", "change-me-in-prod"),
        "BEAUTYAI_API_URL": os.environ.get("BEAUTYAI_API_URL", "http://localhost:8000"),
        "API_VERSION": "v1",
        "FEATURE_FLAGS": {
            "VOICE_STREAMING": True,
        },
    }
