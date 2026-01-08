#!/bin/bash
# Toggle System Prompt Safeguards for Development/Testing
# Created by Lumina Ashley 💕

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env.safeguards"

show_status() {
    if [ -f "$ENV_FILE" ]; then
        source "$ENV_FILE"
        if [ "$DISABLE_SYSTEM_PROMPT_SAFEGUARDS" = "1" ]; then
            echo "🔓 SAFEGUARDS: DISABLED (Developer Mode)"
            echo "   You can chat about anything, not limited to Kesay Clinics topics"
        else
            echo "🔒 SAFEGUARDS: ENABLED (Production Mode)"
            echo "   Kesay Clinics restrictions active"
        fi
    else
        echo "🔒 SAFEGUARDS: ENABLED (Production Mode)"
        echo "   Kesay Clinics restrictions active"
    fi
}

enable_safeguards() {
    if [ -f "$ENV_FILE" ]; then
        rm "$ENV_FILE"
    fi
    echo "✅ Safeguards ENABLED"
    echo "   Model will follow Kesay Clinics restrictions"
    echo "   Arabic-only responses enforced"
    echo ""
    echo "💡 Restart API service for changes to take effect:"
    echo "   sudo systemctl restart beautyai-api.service"
}

disable_safeguards() {
    echo "export DISABLE_SYSTEM_PROMPT_SAFEGUARDS=1" > "$ENV_FILE"
    echo "🔓 Safeguards DISABLED (Developer Mode)"
    echo "   Model will use default medical prompts"
    echo "   No topic restrictions"
    echo "   Language detection works normally"
    echo ""
    echo "⚠️  WARNING: This is for TESTING ONLY!"
    echo "   DO NOT use in production with real patients"
    echo ""
    echo "💡 To apply changes:"
    echo "   1. Source the env file: source $ENV_FILE"
    echo "   2. Restart API: sudo systemctl restart beautyai-api.service"
}

case "$1" in
    enable|on)
        enable_safeguards
        ;;
    disable|off)
        disable_safeguards
        ;;
    status)
        show_status
        ;;
    *)
        echo "🔐 Kesay Clinics Safeguard Toggle"
        echo ""
        echo "Usage: $0 {enable|disable|status}"
        echo ""
        echo "Commands:"
        echo "  enable   - Turn ON Kesay Clinics safeguards (production mode)"
        echo "  disable  - Turn OFF safeguards (developer/testing mode)"
        echo "  status   - Show current safeguard status"
        echo ""
        show_status
        exit 1
        ;;
esac
