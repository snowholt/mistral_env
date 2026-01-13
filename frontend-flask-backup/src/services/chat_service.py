import aiohttp
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ChatAPIClient:
    """Async client to communicate with BeautyAI backend.

    NOTE: One-off sessions are used via context manager; for higher throughput
    we could introduce a session pool.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            raise RuntimeError("ClientSession not initialized – use async context")
        return self._session

    async def list_models(self) -> List[Dict[str, Any]]:
        try:
            async with self.session.get(f"{self.base_url}/models/") as resp:
                if resp.status != 200:
                    logger.warning("List models failed: %s", resp.status)
                    return []
                data = await resp.json()
                return data.get("models", [])
        except Exception as e:  # pragma: no cover (network issues)
            logger.error("Error listing models: %s", e)
            return []

    async def load_model(self, model_name: str) -> Dict[str, Any]:
        try:
            async with self.session.post(
                f"{self.base_url}/models/{model_name}/load", json={"force_reload": False}
            ) as resp:
                return await resp.json()
        except Exception as e:  # pragma: no cover
            return {"success": False, "error": str(e)}

    async def unload_model(self, model_name: str) -> Dict[str, Any]:
        try:
            async with self.session.post(
                f"{self.base_url}/models/{model_name}/unload"
            ) as resp:
                return await resp.json()
        except Exception as e:  # pragma: no cover
            return {"success": False, "error": str(e)}

    async def model_status(self, model_name: str) -> Dict[str, Any]:
        try:
            async with self.session.get(
                f"{self.base_url}/models/{model_name}/status"
            ) as resp:
                if resp.status != 200:
                    return {"status": "unknown", "loaded": False}
                return await resp.json()
        except Exception as e:  # pragma: no cover
            logger.error("Model status error: %s", e)
            return {"status": "error", "loaded": False}

    async def send_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with self.session.post(
                f"{self.base_url}/inference/chat", json=payload
            ) as resp:
                txt = await resp.text()
                if resp.status != 200:
                    logger.error("Chat API error %s: %s", resp.status, txt[:300])
                    return {"success": False, "error": f"{resp.status}: {txt}"}
                try:
                    return await resp.json()
                except Exception as je:  # pragma: no cover
                    return {"success": False, "error": f"Invalid JSON: {je}"}
        except Exception as e:  # pragma: no cover
            logger.error("Chat request failed: %s", e)
            return {"success": False, "error": str(e)}
