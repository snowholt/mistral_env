from flask import jsonify
from typing import Any, Dict

def success(data: Any = None, **extra):
    payload: Dict[str, Any] = {"success": True}
    if data is not None:
        payload["data"] = data
    payload.update(extra)
    return jsonify(payload)

def error(message: str, status: int = 400, **extra):
    payload: Dict[str, Any] = {"success": False, "error": message}
    payload.update(extra)
    return jsonify(payload), status
