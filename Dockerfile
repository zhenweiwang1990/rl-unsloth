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
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .
# Use BuildKit cache mount to speed up pip installs across builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/outputs /workspace/data

# Set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH
# Use system cache directories (will be mounted from host)
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_CACHE=/root/.cache/huggingface/hub

# Make scripts executable
RUN chmod +x /workspace/scripts/*.sh

# Default command: setup and show instructions
CMD ["bash", "-c", "echo '=== Email Agent GRPO Training ==='; echo ''; echo 'Available commands:'; echo '  - Generate database: python -m email_agent.data.local_email_db'; echo '  - Run training: python train_grpo.py'; echo '  - Run evaluation: python eval.py'; echo '  - Test setup: python test_setup.py'; echo ''; exec bash"]
