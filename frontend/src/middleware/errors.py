from flask import jsonify

def register_error_handlers(app):
    @app.errorhandler(404)
    def _not_found(e):  # pragma: no cover
        return jsonify({"success": False, "error": "Not Found"}), 404

    @app.errorhandler(500)
    def _server_error(e):  # pragma: no cover
        return jsonify({"success": False, "error": "Internal Server Error"}), 500
