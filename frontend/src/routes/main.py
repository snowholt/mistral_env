from flask import Blueprint, render_template, session
import uuid

main_bp = Blueprint("main", __name__)

@main_bp.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("chat_ui.html")
