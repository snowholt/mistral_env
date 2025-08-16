from flask import Blueprint, jsonify
from config.constants import BACKEND_SIMPLE_VOICE_WS, WEBSOCKET_SIMPLE_VOICE_PATH

ws_bp = Blueprint("websocket", __name__)

@ws_bp.route(WEBSOCKET_SIMPLE_VOICE_PATH, methods=["GET"])  # Informational endpoint
def simple_voice_info():
    return jsonify({
        "websocket_url": BACKEND_SIMPLE_VOICE_WS,
        "description": "Connect directly to backend for real-time PCM streaming.",
    })
