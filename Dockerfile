FROM nvcr.io/nvidia/pytorch:25.09-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy requirements first
COPY requirements.txt .

# 1. Install Unsloth and Transformers FIRST (just like manual test)
# We force upgrade to ensure no conflicts with pre-installed packages
RUN pip3 install --no-cache-dir --upgrade \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    "transformers>=4.40.0" \
    "torch>=2.0.0"

# 2. Install remaining dependencies
# Remove deprecated pynvml and install nvidia-ml-py to fix warnings
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 uninstall -y pynvml 2>/dev/null || true

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/outputs /workspace/link_search_agent/data

# Set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH
# Use system cache directories (will be mounted from host)
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_CACHE=/root/.cache/huggingface/hub
# ModelScope mirror for China
ENV MODELSCOPE_CACHE=/root/.cache/modelscope

# Make scripts executable
RUN chmod +x /workspace/scripts/*.sh

# Default command: setup and show instructions
CMD ["bash", "-c", "echo '=== Link Search Agent GRPO Training ==='; echo ''; echo 'Available commands:'; echo '  - Run training: python train_grpo_linksearch.py --mode masked'; echo '  - Run evaluation: python eval_linksearch.py'; echo '  - Run benchmark: python benchmark_linksearch.py'; echo '  - Test setup: python test_setup.py'; echo ''; exec bash"]
