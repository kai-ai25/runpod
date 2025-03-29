#!/bin/bash

# --- Configuration ---
TMUX_API_SESSION="mistral_api"
TMUX_UI_SESSION="mistral_ui"
LOG_FILE="/workspace/pod.log"

# --- Stop Services ---
echo "🛑 Stopping Mistral-7B services..."
tmux kill-session -t "$TMUX_API_SESSION" 2>/dev/null
tmux kill-session -t "$TMUX_UI_SESSION" 2>/dev/null
pkill -f "uvicorn" 2>/dev/null
pkill -f "gradio" 2>/dev/null

# --- Cleanup ---
echo "🧹 Cleaning up..."
[ -f "$LOG_FILE" ] && > "$LOG_FILE"

echo -e "\n✅ All services stopped!"
