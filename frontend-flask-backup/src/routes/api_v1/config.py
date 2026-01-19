from flask import Blueprint, current_app, jsonify

config_bp = Blueprint("config_v1", __name__)

@config_bp.route("/config", methods=["GET"])
def get_config():
    # Expose a minimal safe subset for the frontend
    public_cfg = {
        "environment": current_app.config.get("ENVIRONMENT"),
        "default_model": current_app.config.get("DEFAULT_MODEL"),
        "api_base_url": current_app.config.get("BEAUTYAI_API_URL"),
        "version": "v1",
    }
    return jsonify(public_cfg)
