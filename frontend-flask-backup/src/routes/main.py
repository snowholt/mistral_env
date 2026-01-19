from flask import Blueprint, render_template, session, request, abort
import uuid
import os
from dotenv import load_dotenv

# Load environment variables from .env.production if it exists
if os.path.exists('/home/lumi/beautyai/.env.production'):
    load_dotenv('/home/lumi/beautyai/.env.production')
    print("✅ Loaded production environment variables")
else:
    print("⚠️ Production environment file not found")

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
    import time
    
    # Log configuration for debugging
    print(f"🔧 Voice route config: {CONFIG_DICT.get('backend', {})}")
    print(f"🌐 WebSocket URL: {CONFIG_DICT.get('backend', {}).get('simple_voice_ws', 'NOT SET')}")
    print(f"🏭 Environment: {CONFIG_DICT.get('environment', 'unknown')}")
    print(f"🔒 Production: {CONFIG_DICT.get('production', False)}")
    
    return render_template("simple_voice_ui.html", 
                         config=CONFIG_DICT,
                         session_id=session["session_id"],
                         cache_buster=int(time.time()))

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
    import time
    
    return render_template("debug_simple_voice.html",
                         config=CONFIG_DICT,
                         session_id=session["session_id"],
                         cache_buster=int(time.time()))

@main_bp.route("/chat")
def chat():
    """Alternative route for chat interface"""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("chat_ui.html")
