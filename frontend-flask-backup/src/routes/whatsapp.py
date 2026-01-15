"""
WhatsApp Manager routes.

Provides web UI pages for WhatsApp automation management:
- Login/Register
- Onboarding (Meta signup)
- Agent setup
- Inbox (manual chat)
"""
from flask import Blueprint, render_template, redirect, url_for, request, session, flash, jsonify
import os
import requests
import logging

logger = logging.getLogger(__name__)

whatsapp_bp = Blueprint("whatsapp", __name__, url_prefix="/whatsapp")

# Backend API base URL
API_BASE = os.getenv("BACKEND_API_URL", "http://localhost:8000")


def get_auth_headers():
    """Get authorization headers from session."""
    token = session.get("access_token")
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


def is_authenticated():
    """Check if user is authenticated."""
    return session.get("access_token") is not None


# ============================================
# Auth Pages
# ============================================

@whatsapp_bp.route("/login", methods=["GET", "POST"])
def login():
    """Login page."""
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        
        if not email or not password:
            flash("Email and password are required", "error")
            return render_template("whatsapp/login.html")
        
        try:
            # Call backend login API
            response = requests.post(
                f"{API_BASE}/api/v1/auth/login",
                json={"email": email, "password": password},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                session["access_token"] = data["access_token"]
                session["refresh_token"] = data["refresh_token"]
                session["user"] = data.get("user", {})
                flash("Logged in successfully!", "success")
                return redirect(url_for("whatsapp.dashboard"))
            else:
                error = response.json().get("detail", "Invalid credentials")
                flash(error, "error")
        except requests.RequestException as e:
            logger.error(f"Login request failed: {e}")
            flash("Unable to connect to server", "error")
    
    return render_template("whatsapp/login.html")


@whatsapp_bp.route("/register", methods=["GET", "POST"])
def register():
    """Registration page."""
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        full_name = request.form.get("full_name", "").strip()
        business_name = request.form.get("business_name", "").strip()
        
        # Validation
        if not all([email, password, full_name]):
            flash("All fields are required", "error")
            return render_template("whatsapp/register.html")
        
        if password != confirm_password:
            flash("Passwords do not match", "error")
            return render_template("whatsapp/register.html")
        
        if len(password) < 8:
            flash("Password must be at least 8 characters", "error")
            return render_template("whatsapp/register.html")
        
        try:
            # Call backend register API
            response = requests.post(
                f"{API_BASE}/api/v1/auth/register",
                json={
                    "email": email,
                    "password": password,
                    "full_name": full_name,
                    "business_name": business_name or None
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                session["access_token"] = data["access_token"]
                session["refresh_token"] = data["refresh_token"]
                session["user"] = {"id": data["user_id"], "email": email, "full_name": full_name}
                flash("Registration successful! Welcome aboard!", "success")
                return redirect(url_for("whatsapp.onboarding"))
            else:
                error = response.json().get("detail", "Registration failed")
                flash(error, "error")
        except requests.RequestException as e:
            logger.error(f"Register request failed: {e}")
            flash("Unable to connect to server", "error")
    
    return render_template("whatsapp/register.html")


@whatsapp_bp.route("/logout")
def logout():
    """Logout and clear session."""
    session.clear()
    flash("You have been logged out", "info")
    return redirect(url_for("whatsapp.login"))


# ============================================
# Dashboard
# ============================================

@whatsapp_bp.route("/")
@whatsapp_bp.route("/dashboard")
def dashboard():
    """Main dashboard page."""
    if not is_authenticated():
        return redirect(url_for("whatsapp.login"))
    
    # Fetch customers
    customers = []
    try:
        response = requests.get(
            f"{API_BASE}/api/v1/whatsapp/customers",
            headers=get_auth_headers(),
            timeout=10
        )
        if response.status_code == 200:
            customers = response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch customers: {e}")
    
    return render_template(
        "whatsapp/dashboard.html",
        user=session.get("user", {}),
        customers=customers
    )


# ============================================
# Onboarding (Meta Embedded Signup)
# ============================================

@whatsapp_bp.route("/onboarding")
def onboarding():
    """WhatsApp onboarding page with Meta Embedded Signup."""
    if not is_authenticated():
        return redirect(url_for("whatsapp.login"))
    
    # Get Meta app config
    meta_config = {
        "app_id": os.getenv("META_APP_ID", ""),
        "config_id": os.getenv("META_CONFIG_ID", "")
    }
    
    return render_template(
        "whatsapp/onboarding.html",
        user=session.get("user", {}),
        meta_config=meta_config
    )


@whatsapp_bp.route("/onboarding/callback")
def onboarding_callback():
    """Handle Meta OAuth callback."""
    if not is_authenticated():
        return redirect(url_for("whatsapp.login"))
    
    code = request.args.get("code")
    customer_id = request.args.get("customer_id")
    
    if not code:
        flash("Authorization failed - no code received", "error")
        return redirect(url_for("whatsapp.onboarding"))
    
    try:
        # Complete signup with backend
        response = requests.post(
            f"{API_BASE}/api/v1/whatsapp/meta/complete-signup",
            headers=get_auth_headers(),
            json={"customer_id": int(customer_id), "code": code},
            timeout=30
        )
        
        if response.status_code == 200:
            flash("WhatsApp account connected successfully!", "success")
            return redirect(url_for("whatsapp.agent_setup"))
        else:
            error = response.json().get("detail", "Failed to connect WhatsApp")
            flash(error, "error")
    except requests.RequestException as e:
        logger.error(f"Callback request failed: {e}")
        flash("Unable to connect to server", "error")
    
    return redirect(url_for("whatsapp.onboarding"))


# ============================================
# Agent Setup
# ============================================

@whatsapp_bp.route("/agent-setup", methods=["GET", "POST"])
def agent_setup():
    """Agent configuration page."""
    if not is_authenticated():
        return redirect(url_for("whatsapp.login"))
    
    # Get customer ID from query or session
    customer_id = request.args.get("customer_id") or session.get("current_customer_id")
    
    if request.method == "POST":
        customer_id = request.form.get("customer_id")
        business_name = request.form.get("business_name", "").strip()
        tone = request.form.get("tone", "professional")
        behavior_rules = request.form.get("behavior_rules", "").strip()
        custom_instructions = request.form.get("custom_instructions", "").strip()
        
        if not business_name:
            flash("Business name is required", "error")
            return render_template("whatsapp/agent_setup.html", user=session.get("user", {}))
        
        try:
            response = requests.post(
                f"{API_BASE}/api/v1/whatsapp/agents/configure",
                headers=get_auth_headers(),
                json={
                    "customer_id": int(customer_id),
                    "business_name": business_name,
                    "tone": tone,
                    "behavior_rules": behavior_rules or None,
                    "custom_instructions": custom_instructions or None
                },
                timeout=10
            )
            
            if response.status_code == 200:
                flash("AI agent configured successfully!", "success")
                return redirect(url_for("whatsapp.inbox"))
            else:
                error = response.json().get("detail", "Configuration failed")
                flash(error, "error")
        except requests.RequestException as e:
            logger.error(f"Agent config request failed: {e}")
            flash("Unable to connect to server", "error")
    
    # Fetch existing config if available
    existing_config = None
    if customer_id:
        try:
            response = requests.get(
                f"{API_BASE}/api/v1/whatsapp/agents/config/{customer_id}",
                headers=get_auth_headers(),
                timeout=10
            )
            if response.status_code == 200:
                existing_config = response.json()
        except requests.RequestException:
            pass
    
    # Fetch customers for dropdown
    customers = []
    try:
        response = requests.get(
            f"{API_BASE}/api/v1/whatsapp/customers",
            headers=get_auth_headers(),
            timeout=10
        )
        if response.status_code == 200:
            customers = response.json()
    except requests.RequestException:
        pass
    
    return render_template(
        "whatsapp/agent_setup.html",
        user=session.get("user", {}),
        customers=customers,
        existing_config=existing_config,
        customer_id=customer_id
    )


# ============================================
# Inbox
# ============================================

@whatsapp_bp.route("/inbox")
def inbox():
    """Chat inbox page with real-time WebSocket updates."""
    if not is_authenticated():
        return redirect(url_for("whatsapp.login"))
    
    # Get access token for WebSocket
    access_token = session.get("access_token", "")
    
    # WebSocket URL - construct full path with token
    ws_base = os.getenv("WS_URL", "ws://localhost:8000")
    ws_url = f"{ws_base}/api/v1/whatsapp/inbox/ws?token={access_token}"
    
    # Fetch customers for selector
    customers = []
    customer_id = None
    try:
        response = requests.get(
            f"{API_BASE}/api/v1/whatsapp/customers",
            headers=get_auth_headers(),
            timeout=10
        )
        if response.status_code == 200:
            customers = response.json()
            if customers:
                customer_id = customers[0].get("id")
    except requests.RequestException as e:
        logger.error(f"Failed to fetch customers for inbox: {e}")
    
    return render_template(
        "whatsapp/inbox.html",
        user=session.get("user", {}),
        customers=customers,
        customer_id=customer_id,
        ws_url=ws_url,
        api_base=API_BASE
    )


# ============================================
# API Proxies (for frontend fetch calls)
# ============================================

@whatsapp_bp.route("/api/conversations")
def api_conversations():
    """Proxy for fetching conversations."""
    if not is_authenticated():
        return jsonify({"error": "Unauthorized"}), 401
    
    customer_id = request.args.get("customer_id")
    
    try:
        params = {}
        if customer_id:
            params["customer_id"] = customer_id
        
        response = requests.get(
            f"{API_BASE}/api/v1/whatsapp/inbox/conversations",
            headers=get_auth_headers(),
            params=params,
            timeout=10
        )
        return jsonify(response.json()), response.status_code
    except requests.RequestException as e:
        logger.error(f"Conversations fetch failed: {e}")
        return jsonify({"error": "Server error"}), 500


@whatsapp_bp.route("/api/conversations/<int:conversation_id>/messages")
def api_messages(conversation_id):
    """Proxy for fetching messages."""
    if not is_authenticated():
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        response = requests.get(
            f"{API_BASE}/api/v1/whatsapp/inbox/conversations/{conversation_id}/messages",
            headers=get_auth_headers(),
            timeout=10
        )
        return jsonify(response.json()), response.status_code
    except requests.RequestException as e:
        logger.error(f"Messages fetch failed: {e}")
        return jsonify({"error": "Server error"}), 500


@whatsapp_bp.route("/api/conversations/<int:conversation_id>/messages", methods=["POST"])
def api_send_message(conversation_id):
    """Proxy for sending messages."""
    if not is_authenticated():
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        response = requests.post(
            f"{API_BASE}/api/v1/whatsapp/inbox/conversations/{conversation_id}/messages",
            headers=get_auth_headers(),
            json=request.json,
            timeout=30
        )
        return jsonify(response.json()), response.status_code
    except requests.RequestException as e:
        logger.error(f"Send message failed: {e}")
        return jsonify({"error": "Server error"}), 500
