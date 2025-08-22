#!/usr/bin/env python3
"""Minimal script to manually inspect API vs Direct chat responses.

Usage:
  python tests/test_api_direct_mode_consistency.py \
      --model qwen3-unsloth-q4ks \
      --message "/no_think What is botox?" \
      --temperature 0.0 --max_new_tokens 200 --do_sample False 

This script intentionally performs NO automatic comparison, regex cleanup,
benchmarking, or assertions. It only prints:
  1. Raw API JSON response
  2. Raw Direct Chat response dict

You can visually compare:
  - Are they the same textual answer?
  - Is the output structure/format similar?
  - (Optionally) Inspect whether the same underlying model instance appears to be reused.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import requests

# Silence noisy loggers (only show critical errors if any)
logging.basicConfig(level=logging.CRITICAL)
for noisy in [
    "uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "transformers", "torch"
]:
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

# Add backend source path for direct imports
backend_src = Path(__file__).parent.parent / "backend" / "src"
sys.path.insert(0, str(backend_src))

from beautyai_inference.services.inference.chat_service import ChatService  # type: ignore
from beautyai_inference.config.config_manager import AppConfig  # type: ignore
from beautyai_inference.services.model.registry_service import ModelRegistryService  # type: ignore


API_BASE_URL_DEFAULT = "http://127.0.0.1:8000"

# Global single ChatService instance to let you inspect whether reused
_CHAT_SERVICE = ChatService()
_APP_CONFIG = AppConfig()
_MODEL_REGISTRY_SERVICE = ModelRegistryService()


def api_chat(api_base_url: str, model_name: str, message: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Call the REST API chat endpoint and return its raw JSON response."""
    payload = {"model_name": model_name, "message": message}
    payload.update(params)
    resp = requests.post(f"{api_base_url}/inference/chat", json=payload, timeout=60)
    try:
        data = resp.json()
    except Exception:
        data = {"error": True, "status_code": resp.status_code, "text": resp.text}
    return data


def direct_chat(model_name: str, message: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Call the ChatService directly and return a simple dict of raw pieces.

    Returns keys similar in spirit to API for easy manual comparison.
    """
    # Ensure model registry loaded (if not already)
    if not getattr(_APP_CONFIG, "model_registry", None):
        _APP_CONFIG.load_model_registry()

    model_config = _MODEL_REGISTRY_SERVICE.get_model(_APP_CONFIG, model_name)
    if model_config is None:
        return {"error": f"Model '{model_name}' not found in registry"}

    # Generation config: remove API-only flags that ChatService.chat may not expect
    generation_config = params.copy()
    generation_config.pop("disable_content_filter", None)
    generation_config.pop("enable_thinking", None)

    response, detected_language, _, session_id = _CHAT_SERVICE.chat(
        message=message,
        model_name=model_name,
        model_config=model_config,
        generation_config=generation_config,
        conversation_history=[],
        response_language="auto",
        session_id=None,
        disable_content_filter=params.get("disable_content_filter", False),
    )

    return {
        "final_content": response,
        "detected_language": detected_language,
        "session_id": session_id,
        "chat_service_id": id(_CHAT_SERVICE),  # so you can see instance identity
        "model_name": model_name,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal API vs Direct chat inspection tool")
    p.add_argument("--api-base-url", default=API_BASE_URL_DEFAULT, help="API base URL")
    p.add_argument("--model", required=True, help="Model name as in model registry")
    p.add_argument("--message", required=True, help="User message/prompt (can include /no_think)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--do_sample", type=str, default="False", help="True/False")
    p.add_argument("--disable_content_filter", type=str, default="True", help="True/False")
    p.add_argument("--enable_thinking", type=str, default="False", help="True/False")
    return p.parse_args()


def str2bool(v: str) -> bool:
    return v.lower() in {"1", "true", "yes", "y", "on"}


def main() -> int:
    args = parse_args()

    params: Dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": str2bool(args.do_sample),
        "disable_content_filter": str2bool(args.disable_content_filter),
        "enable_thinking": str2bool(args.enable_thinking),
    }

    print("=== API CALL ===")
    api_res = api_chat(args.api_base_url, args.model, args.message, params)
    print(json.dumps(api_res, ensure_ascii=False, indent=2))

    print("\n=== DIRECT CALL ===")
    direct_res = direct_chat(args.model, args.message, params)
    print(json.dumps(direct_res, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())