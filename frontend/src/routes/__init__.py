from flask import Blueprint
from .main import main_bp
from .websocket import ws_bp
from .debug import debug_bp
from .api_v1.chat import chat_bp
from .api_v1.models import models_bp
from .api_v1.config import config_bp


def register_blueprints(app):
    app.register_blueprint(main_bp)
    app.register_blueprint(ws_bp)
    app.register_blueprint(debug_bp, url_prefix="/debug")
    app.register_blueprint(chat_bp, url_prefix="/api/v1")
    app.register_blueprint(models_bp, url_prefix="/api/v1")
    app.register_blueprint(config_bp, url_prefix="/api/v1")
