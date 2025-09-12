from flask import Blueprint, render_template, session, request, abort
import uuid

main_bp = Blueprint("main", __name__)

@main_bp.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("chat_ui.html")

@main_bp.route("/voice")
def voice():
    """Main voice interface route"""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    
    # Get configuration from app config or environment
    from config.constants import CONFIG_DICT
    
    return render_template("simple_voice_ui.html", 
                         config=CONFIG_DICT,
                         session_id=session["session_id"])

@main_bp.route("/voice/debug")
def voice_debug():
    """Debug interface for voice functionality"""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    
    # Optional: Add access control for debug interface
    # This could check for admin permissions, development environment, etc.
    debug_enabled = request.args.get('enable') == 'true'
    
    # You can add environment checks here:
    # import os
    # if os.getenv('ENVIRONMENT') == 'production' and not debug_enabled:
    #     abort(404)
    
    from config.constants import CONFIG_DICT
    
    return render_template("debug_simple_voice.html",
                         config=CONFIG_DICT,
                         session_id=session["session_id"])

@main_bp.route("/chat")
def chat():
    """Alternative route for chat interface"""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("chat_ui.html")
