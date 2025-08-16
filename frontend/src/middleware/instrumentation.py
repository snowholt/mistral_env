import time
from flask import g, request


def register_request_instrumentation(app):
    @app.before_request
    def _start_timer():  # pragma: no cover simple
        g._start_time = time.time()

    @app.after_request
    def _log_timing(response):  # pragma: no cover
        start = getattr(g, "_start_time", None)
        if start:
            duration_ms = (time.time() - start) * 1000
            app.logger.debug(
                "%s %s -> %s (%.2f ms)", request.method, request.path, response.status_code, duration_ms
            )
            response.headers["X-Request-Duration-ms"] = f"{duration_ms:.2f}"
        return response
