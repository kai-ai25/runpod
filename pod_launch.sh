#!/bin/bash
set -euo pipefail

# --- Configuration ---
HF_TOKEN=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  # Replace with your actual token
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.1"
LOCAL_MODEL_DIR="/workspace/huggingface/mistral-7b-instruct"
API_PORT=8000
UI_PORT=7860
LOG_FILE="/workspace/log_pod.log"
TMUX_API_SESSION="mistral_api"
TMUX_UI_SESSION="mistral_ui"
MIN_DISK_SPACE_MB=30000  # 30GB minimum

# --- Error Handler ---
handle_error() {
    echo "❌ Error at line $1" | tee -a "$LOG_FILE"
    echo "💡 Check logs: tail -f $LOG_FILE" | tee -a "$LOG_FILE"
    # Continue execution despite errors
}
trap 'handle_error $LINENO' ERR

# --- Initialize ---
mkdir -p "$LOCAL_MODEL_DIR"
touch "$LOG_FILE"
export HF_TOKEN="$HF_TOKEN"

# --- Cleanup Previous Runs ---
echo "🧹 Cleaning up previous sessions..." | tee -a "$LOG_FILE"
{
    pkill -f "uvicorn.*$API_PORT" || true
    pkill -f "gradio.*$UI_PORT" || true
    tmux kill-session -t "$TMUX_API_SESSION" 2>/dev/null || true
    tmux kill-session -t "$TMUX_UI_SESSION" 2>/dev/null || true
} >> "$LOG_FILE" 2>&1

# --- System Checks ---
verify_resources() {
    echo "🔍 Verifying system resources..." | tee -a "$LOG_FILE"
    
    local available_mb=$(df -m "$LOCAL_MODEL_DIR" | awk 'NR==2 {print $4}')
    if [ "$available_mb" -lt "$MIN_DISK_SPACE_MB" ]; then
        echo "❌ Insufficient disk space (${available_mb}MB available, ${MIN_DISK_SPACE_MB}MB required)" | tee -a "$LOG_FILE"
        exit 1
    fi

    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected" | tee -a "$LOG_FILE"
    else
        echo "⚠️ No NVIDIA GPU detected - will use CPU" | tee -a "$LOG_FILE"
    fi
}

# --- Dependency Installation ---
install_dependencies() {
    echo "📦 Installing dependencies..." | tee -a "$LOG_FILE"
    
    {
        apt-get update -qq && apt-get install -y \
            python3 python3-pip python3-dev \
            tmux curl git git-lfs net-tools jq \
            nvidia-cuda-toolkit
        
        pip install --upgrade pip
        pip install \
            torch transformers accelerate bitsandbytes \
            fastapi uvicorn gradio huggingface-hub \
            tqdm  # For progress bars
    } >> "$LOG_FILE" 2>&1
}

# --- Model Download with Progress Bar ---
download_model() {
    echo "🔐 Authenticating and downloading model..." | tee -a "$LOG_FILE"
    
    # Write token to cache file
    mkdir -p ~/.cache/huggingface
    echo "$HF_TOKEN" > ~/.cache/huggingface/token
    
    # Verify token
    if ! huggingface-cli whoami >> "$LOG_FILE" 2>&1; then
        echo "❌ Invalid Hugging Face token - please verify and update HF_TOKEN" | tee -a "$LOG_FILE"
        exit 1
    fi

    # Verify model access
    if ! curl -s -H "Authorization: Bearer $HF_TOKEN" \
        "https://huggingface.co/api/models/$MODEL_NAME" >> "$LOG_FILE" 2>&1; then
        echo "❌ Cannot access model - ensure you've accepted the license at:" | tee -a "$LOG_FILE"
        echo "   https://huggingface.co/$MODEL_NAME" | tee -a "$LOG_FILE"
        exit 1
    fi

    local retries=3
    local wait_time=30
    
    for ((i=1; i<=retries; i++)); do
        echo "⬇️ Download attempt $i/$retries..." | tee -a "$LOG_FILE"
        
        # Python script with progress bar
        if python3 - <<EOF 2>&1 | tee -a "$LOG_FILE"
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
import os

class TqdmProgress:
    def __init__(self):
        self.pbar = None
    
    def __call__(self, **kwargs):
        if self.pbar is None:
            self.pbar = tqdm(
                total=kwargs.get("total", 0),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading model"
            )
        self.pbar.update(kwargs.get("advance", 0))

try:
    snapshot_download(
        repo_id="$MODEL_NAME",
        local_dir="$LOCAL_MODEL_DIR",
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,
        token=os.environ.get("HF_TOKEN"),
        callback=TqdmProgress()
    )
    print("✅ Download successful!")
except Exception as e:
    print(f"❌ Download failed: {str(e)}")
    exit(1)
EOF
        then
            return 0
        fi
        
        echo "⚠️ Attempt failed, retrying in $wait_time seconds..." | tee -a "$LOG_FILE"
        sleep $wait_time
    done
    
    echo "❌ All download attempts failed" | tee -a "$LOG_FILE"
    exit 1
}

# --- Service Setup ---
setup_services() {
    echo "🚀 Setting up services..." | tee -a "$LOG_FILE"
    
    # API Server
    cat <<EOT > api_server.py
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
import os

app = FastAPI()

print("⚙️ Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "$LOCAL_MODEL_DIR",
    device_map="auto",
    torch_dtype=torch.float16,
    token=os.environ.get("HF_TOKEN"))
tokenizer = AutoTokenizer.from_pretrained("$LOCAL_MODEL_DIR")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/generate")
def generate(prompt: str, max_tokens: int = 100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return {"response": tokenizer.decode(outputs[0], skip_special_tokens=True)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=$API_PORT)
EOT

    # Gradio UI
    cat <<EOT > gradio_ui.py
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

print("⚙️ Loading model for UI...")
model = AutoModelForCausalLM.from_pretrained(
    "$LOCAL_MODEL_DIR",
    device_map="auto",
    torch_dtype=torch.float16,
    token=os.environ.get("HF_TOKEN"))
tokenizer = AutoTokenizer.from_pretrained("$LOCAL_MODEL_DIR")

def respond(message, history):
    inputs = tokenizer(message, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

gr.ChatInterface(respond).launch(
    server_name="0.0.0.0",
    server_port=$UI_PORT,
    share=False
)
EOT
}

# --- Service Launch ---
launch_services() {
    echo "🚀 Starting services..." | tee -a "$LOG_FILE"
    
    # API Service
    if ! tmux new-session -d -s "$TMUX_API_SESSION" "python api_server.py >> $LOG_FILE 2>&1"; then
        echo "⚠️ Failed to start API service (check logs)" | tee -a "$LOG_FILE"
    else
        echo "✅ API service started in tmux session: $TMUX_API_SESSION" | tee -a "$LOG_FILE"
    fi
    
    # UI Service
    if ! tmux new-session -d -s "$TMUX_UI_SESSION" "sleep 15 && python gradio_ui.py >> $LOG_FILE 2>&1"; then
        echo "⚠️ Failed to start UI service (check logs)" | tee -a "$LOG_FILE"
    else
        echo "✅ UI service started in tmux session: $TMUX_UI_SESSION" | tee -a "$LOG_FILE"
    fi
    
    # Verify services (non-critical)
    sleep 15
    if curl -s "http://localhost:$API_PORT/health" | grep -q "ok"; then
        echo "✅ API health check passed" | tee -a "$LOG_FILE"
    else
        echo "⚠️ API health check failed (service may still start later)" | tee -a "$LOG_FILE"
    fi
}

# --- Main Execution ---
{
    echo -e "\n=== Mistral-7B Deployment ==="
    verify_resources
    install_dependencies
    download_model
    setup_services
    launch_services

    echo -e "\n✅ Deployment completed!"
    echo "📌 API:    http://localhost:$API_PORT/docs" 
    echo "📌 UI:     http://localhost:$UI_PORT"
    echo "📌 Logs:   tail -f $LOG_FILE"
    echo -e "\n🛑 To stop: tmux kill-session -t $TMUX_API_SESSION && tmux kill-session -t $TMUX_UI_SESSION"
} | tee -a "$LOG_FILE"

# Exit successfully
exit 0
