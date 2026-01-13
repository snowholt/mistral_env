from flask import Blueprint, current_app, jsonify
from services.chat_service import ChatAPIClient
from utils.aio import run_async

models_bp = Blueprint("models_v1", __name__)

@models_bp.route("/models", methods=["GET"])
def list_models():
    async def _get():
        async with ChatAPIClient(current_app.config["BEAUTYAI_API_URL"]) as client:
            return await client.list_models()

    result = run_async(_get())
    return jsonify({"success": True, "models": result})
