from flask import Blueprint, render_template

debug_bp = Blueprint("debug", __name__, template_folder="../templates")

@debug_bp.route("/pcm-upload")
def pcm_upload():
    return render_template("debug_pcm_upload.html")

@debug_bp.route("/streaming-live")
def streaming_live():
    return render_template("debug_streaming_live.html")
