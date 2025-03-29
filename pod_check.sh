#!/bin/bash

# --- Configuration ---
API_PORT=8000
UI_PORT=7860
TMUX_API_SESSION="mistral_api"
TMUX_UI_SESSION="mistral_ui"
LOG_FILE="/workspace/pod.log"

# --- Status Checks ---
check_service() {
    local name=$1
    local port=$2
    local session=$3
    
    # Check port
    if netstat -tuln | grep -q ":$port "; then
        port_status="✅"
    else
        port_status="❌"
    fi
    
    # Check tmux
    if tmux has-session -t "$session" 2>/dev/null; then
        tmux_status="✅"
    else
        tmux_status="❌"
    fi
    
    printf "%-10s %-2s (Port) | %-2s (Tmux)\n" "$name:" "$port_status" "$tmux_status"
}

# --- Display Status ---
echo -e "\n🔍 Mistral-7B Service Status"
echo "-----------------------------------"
check_service "API" "$API_PORT" "$TMUX_API_SESSION"
check_service "UI" "$UI_PORT" "$TMUX_UI_SESSION"
echo "-----------------------------------"

# --- Log Info ---
echo -e "\n📜 Last 5 log lines:"
tail -n 5 "$LOG_FILE" 2>/dev/null || echo "No log file found"
