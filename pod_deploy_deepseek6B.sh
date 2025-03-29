#!/bin/bash
set -euo pipefail

# --- Configuration ---
HF_TOKEN=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  # Replace with your actual token
MODEL_NAME="deepseek-ai/deepseek-coder-6.7b-instruct"
LOCAL_MODEL_DIR="/workspace/huggingface/deepseek-6.7b"
API_PORT=8000
UI_PORT=7860
LOG_FILE="/workspace/log_deepseek.log"
TMUX_API_SESSION="deepseek_api"
TMUX_UI_SESSION="deepseek_ui"
MIN_DISK_SPACE_MB=30000  # 30GB minimum (6.7B model is smaller)

# --- Error Handler ---
handle_error() {
    echo "❌ Error at line $1" | tee -a "$LOG_FILE"
    echo "💡 Check logs: tail -f $LOG_FILE" | tee -a "$LOG_FILE"
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
        # Check GPU memory (6.7B needs ~16GB in FP16, ~8GB in 4-bit)
        local gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        if [ "$gpu_mem" -lt 16000 ]; then
            echo "⚠️ Enabling 4-bit quantization for lower VRAM usage" | tee -a "$LOG_FILE"
            export USE_4BIT="true"
        fi
    else
        echo "⚠️ No NVIDIA GPU detected - will use CPU (slow for 6.7B!)" | tee -a "$LOG_FILE"
    fi
}

# --- Dependency Installation ---
install_dependencies() {
    echo "📦 Installing dependencies..." | tee -a "$LOG_FILE"
    
    {
        apt-get update -qq && apt-get install -y \
            python3 python3-pip python3-dev \
            tmux curl git git-lfs net-tools jq
        
        # Only install CUDA toolkit if GPU is detected
        if command -v nvidia-smi &> /dev/null; then
            apt-get install -y nvidia-cuda-toolkit
        fi
        
        pip install --upgrade pip
        pip install \
            torch transformers accelerate bitsandbytes \
            fastapi uvicorn gradio huggingface-hub \
            tqdm auto-gptq
    } >> "$LOG_FILE" 2>&1
}

# --- Model Download ---
download_model() {
    echo "🔐 Authenticating and downloading model..." | tee -a "$LOG_FILE"
    
    if [ -f "$LOCAL_MODEL_DIR/config.json" ]; then
        echo "Model already downloaded. Skipping download." | tee -a "$LOG_FILE"
        return 0
    fi
    
    # Verify token
    if ! huggingface-cli whoami >> "$LOG_FILE" 2>&1; then
        echo "❌ Invalid Hugging Face token - please update HF_TOKEN" | tee -a "$LOG_FILE"
        exit 1
    fi

    # Verify model access
    if ! curl -s -H "Authorization: Bearer $HF_TOKEN" \
        "https://huggingface.co/api/models/$MODEL_NAME" >> "$LOG_FILE" 2>&1; then
        echo "❌ Cannot access model - ensure you've accepted the license at:" | tee -a "$LOG_FILE"
        echo "   https://huggingface.co/$MODEL_NAME" | tee -a "$LOG_FILE"
        exit 1
    fi

    echo "⬇️ Downloading DeepSeek-6.7B (this may take 10-30 mins)..." | tee -a "$LOG_FILE"

    
        # New download method without callback
    python3 - <<EOF 2>&1 | tee -a "$LOG_FILE"
from huggingface_hub import snapshot_download
import os

try:
    snapshot_download(
        repo_id="$MODEL_NAME",
        local_dir="$LOCAL_MODEL_DIR",
        local_dir_use_symlinks=False,
        resume_download=True,
        token=os.environ.get("HF_TOKEN")
    )
    print("✅ Download successful!")
except Exception as e:
    print(f"❌ Download failed: {str(e)}")
    exit(1)
EOF

    # Verify download
    if [ ! -f "$LOCAL_MODEL_DIR/config.json" ]; then
        echo "❌ Model files not found - download may have failed" | tee -a "$LOG_FILE"
        exit 1
    fi
}

# --- Service Setup (Optimized for 6.7B) ---
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

print("⚙️ Loading DeepSeek-6.7B...")
model_kwargs = {
    "device_map": "auto",
    "token": os.environ.get("HF_TOKEN")
}

# Optimize for 6.7B model size
if os.getenv("USE_4BIT") == "true":
    model_kwargs.update({
        "load_in_4bit": True,
        "torch_dtype": torch.float16,
        "quantization_config": {
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True
        }
    })
else:
    model_kwargs.update({
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True
    })

model = AutoModelForCausalLM.from_pretrained(
    "$LOCAL_MODEL_DIR",
    **model_kwargs
)
tokenizer = AutoTokenizer.from_pretrained("$LOCAL_MODEL_DIR")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/generate")
def generate(prompt: str, max_tokens: int = 200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return {"response": tokenizer.decode(outputs[0], skip_special_tokens=True)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=$API_PORT)
EOT

    # Only create default gradio_ui.py if missing
    if [ ! -f "gradio_ui.py" ]; then
        echo "📝 Creating default gradio_ui.py..." | tee -a "$LOG_FILE"
        cat <<'EOT' > gradio_ui.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

model = AutoModelForCausalLM.from_pretrained(
    "$LOCAL_MODEL_DIR",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("$LOCAL_MODEL_DIR")

def respond(message, history):
    inputs = tokenizer(message, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

gr.ChatInterface(respond).launch(server_port=$UI_PORT)

    else
        echo "🔍 Using existing gradio_ui.py (customizations preserved)" | tee -a "$LOG_FILE"
    fi

}

EOT

# --- Service Launch ---
launch_services() {
    echo "🚀 Starting services..." | tee -a "$LOG_FILE"
    
    # API Service
    tmux new-session -d -s "$TMUX_API_SESSION" "python api_server.py >> $LOG_FILE 2>&1"
    echo "✅ API service started in tmux session: $TMUX_API_SESSION" | tee -a "$LOG_FILE"
    
    # UI Service
    tmux new-session -d -s "$TMUX_UI_SESSION" "sleep 10 && python gradio_ui.py >> $LOG_FILE 2>&1"
    echo "✅ UI service started in tmux session: $TMUX_UI_SESSION" | tee -a "$LOG_FILE"
    
    # Verify API
    sleep 10
    if curl -s "http://localhost:$API_PORT/health" | grep -q "ok"; then
        echo "✅ API health check passed" | tee -a "$LOG_FILE"
    else
        echo "⚠️ API health check failed (service may still start later)" | tee -a "$LOG_FILE"
    fi
}

# --- Main Execution ---
{
    echo -e "\n=== DeepSeek-6.7B Deployment ==="
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

exit 0
