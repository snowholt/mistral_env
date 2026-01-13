"""BeautyAI WebUI Flask application entrypoint (minimal).

All application logic lives in subpackages (routes/, services/, middleware/, etc.).
This file intentionally stays tiny and should not accumulate legacy code.
"""
from __future__ import annotations

from flask import Flask

from config.settings import load_config
from routes import register_blueprints
from middleware import register_error_handlers, register_request_instrumentation


def create_app() -> Flask:
    """Create and configure the Flask application instance."""
    app = Flask(__name__)
    app.config.update(load_config())
    register_request_instrumentation(app)
    register_error_handlers(app)
    register_blueprints(app)
    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=5001, debug=True)