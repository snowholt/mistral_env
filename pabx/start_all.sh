#!/bin/bash
# Quick Start Helper Script
# Run this to start both backend and frontend automatically

PABX_DIR="/home/lumi/beautyai/pabx"

echo "🚀 BeautyAI PABX Quick Start"
echo "=============================="
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux not found. Installing..."
    sudo apt install -y tmux
fi

# Create tmux session
SESSION="pabx"

echo "🔧 Starting PABX system in tmux session: $SESSION"
echo ""

# Kill existing session if it exists
tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

# Create new session with backend
tmux new-session -d -s $SESSION -n backend "cd $PABX_DIR && source venv/bin/activate && ./run_server.py --mode api"

# Create frontend window
tmux new-window -t $SESSION -n frontend "cd $PABX_DIR/ui && npm run dev"

# Attach to session
echo "✅ PABX system started!"
echo ""
echo "📋 Running in tmux session: $SESSION"
echo ""
echo "🎯 Tmux Commands:"
echo "   Ctrl+B, D    - Detach from session"
echo "   Ctrl+B, N    - Next window"
echo "   Ctrl+B, P    - Previous window"
echo "   Ctrl+B, 0    - Switch to backend window"
echo "   Ctrl+B, 1    - Switch to frontend window"
echo ""
echo "🌐 Access Points:"
echo "   Frontend:  http://localhost:3000"
echo "   API:       http://localhost:8080"
echo "   API Docs:  http://localhost:8080/docs"
echo ""
echo "🛑 To stop: tmux kill-session -t $SESSION"
echo ""
echo "Attaching to session in 3 seconds..."
sleep 3

tmux attach-session -t $SESSION
