from typing import Dict, Any, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

ALLOWED_GENERATION_PARAMS = {"max_tokens", "temperature", "top_p", "repetition_penalty", "top_k", "min_p"}


def normalize_thinking_mode(data: Dict[str, Any]) -> Optional[bool]:
    if "thinking_mode" in data:
        val = data["thinking_mode"]
    elif "enable_thinking" in data:  # legacy
        val = data["enable_thinking"]
    else:
        return None

    if isinstance(val, str):
        lv = val.lower()
        return lv in ("true", "enable", "1", "yes")
    return bool(val)


def build_chat_payload(
    request_json: Dict[str, Any],
    session_id: Optional[str],
    model_info: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Construct backend payload + metadata.

    Returns: (payload, meta)
    meta includes thought about applied transformations.
    """
    model_name = request_json.get("model_name")
    thinking_mode = normalize_thinking_mode(request_json)

    generation_config = {
        "max_tokens": int(
            request_json.get("max_new_tokens", request_json.get("max_tokens", 2048))
        ),
        "temperature": float(request_json.get("temperature", 0.3)),
        "top_p": float(request_json.get("top_p", 0.95)),
        "repetition_penalty": float(request_json.get("repetition_penalty", 1.1)),
    }

    # Optional params
    for opt in ("top_k", "min_p"):
        if opt in request_json:
            generation_config[opt] = (
                int(request_json[opt]) if opt == "top_k" else float(request_json[opt])
            )

    original_message = request_json.get("message", "")
    message_to_send = original_message
    added_no_think = False
    if thinking_mode is False and original_message:
        message_to_send = f"\\no_think {original_message}"
        added_no_think = True

    payload = {
        "model_name": model_name,
        "message": message_to_send,
        "session_id": session_id,
        "chat_history": request_json.get("chat_history", []),
        "generation_config": generation_config,
        "stream": bool(request_json.get("stream", False)),
    }

    if request_json.get("disable_content_filter"):
        payload["disable_content_filter"] = True
    if "content_filter_strictness" in request_json:
        payload["content_filter_strictness"] = request_json["content_filter_strictness"]
    if thinking_mode is not None:
        payload["thinking_mode"] = thinking_mode

    meta = {
        "thinking_mode": thinking_mode,
        "added_no_think": added_no_think,
    }
    return payload, meta
