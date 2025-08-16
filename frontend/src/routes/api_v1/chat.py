from flask import Blueprint, request, session, jsonify, current_app
from services.chat_service import ChatAPIClient
from services.payload_service import build_chat_payload
from utils.aio import run_async
from config.constants import DEFAULT_MODEL

chat_bp = Blueprint("chat_v1", __name__)

@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    if "model_name" not in data:
        data["model_name"] = DEFAULT_MODEL

    # Fetch model info (optional for now – kept for parity)
    async def _get_models():
        async with ChatAPIClient(current_app.config["BEAUTYAI_API_URL"]) as client:
            return await client.list_models()

    models = run_async(_get_models())
    model_info = next(
        (m for m in models if (m.get("name") or m.get("model_name")) == data["model_name"]),
        None,
    )

    payload, meta = build_chat_payload(data, session.get("session_id"), model_info)

    async def _send():
        async with ChatAPIClient(current_app.config["BEAUTYAI_API_URL"]) as client:
            return await client.send_chat(payload)

    result = run_async(_send())

    if not result.get("success"):
        return jsonify({"success": False, "error": result.get("error", "chat failed")}), 500

    generation_stats = result.get("generation_stats", {})
    perf = generation_stats.get("performance", {})

    return jsonify({
        "success": True,
        "response": result.get("response", ""),
        "thinking_content": result.get("thinking_content", ""),
        "final_content": result.get("final_content", ""),
        "tokens_generated": perf.get("tokens_generated", 0),
        "generation_time_ms": perf.get("generation_time_ms", 0),
        "tokens_per_second": perf.get("tokens_per_second", 0),
        "preset_used": result.get("preset_used", ""),
        "content_filter_applied": result.get("content_filter_applied", False),
        "thinking_enabled": result.get("thinking_enabled", False),
        "content_filter_strictness": result.get("content_filter_strictness", "balanced"),
        "meta": meta,
    })
