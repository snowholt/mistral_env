from flask import Blueprint, render_template

debug_bp = Blueprint("debug", __name__, template_folder="../templates")

@debug_bp.route("/voice-websocket-tester")
def voice_websocket_tester():
    return render_template("debug_voice_websocket_tester.html")

@debug_bp.route("/streaming-live")
def streaming_live():
    return render_template("debug_streaming_live.html")
