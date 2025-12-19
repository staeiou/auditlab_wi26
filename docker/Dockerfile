FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------
# System deps + Python + Node (no conda, no nvm)
# --------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev python3-pip \
        build-essential net-tools \
        git curl wget cmake libcurl4-openssl-dev less zip \
        tmux htop nvtop iotop jnettop nano pciutils \
        ca-certificates gnupg \
        libgomp1 libnuma1 libstdc++6 && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Make "python" point to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# --------------------------------------------------------------------
# Install uv globally
# --------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    install -m 0755 /root/.local/bin/uv /usr/local/bin/uv && \
    rm -f /root/.local/bin/uv

# --------------------------------------------------------------------
# Create venv in /opt (safe with K8s emptyDir mounted at /workspace)
# --------------------------------------------------------------------
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1

# --------------------------------------------------------------------
# Install PyTorch for CUDA 12.9 (do this from the cu129 index ONLY)
# --------------------------------------------------------------------
RUN uv pip install --upgrade pip setuptools wheel && \
    uv pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu129 \
      torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
# PyTorch 2.9.0 package set/version refs. :contentReference[oaicite:1]{index=1}

# --------------------------------------------------------------------
# Install vLLM (from PyPI) + your extras (from PyPI)
# vLLM wheels are compiled with CUDA 12.9 by default; PyTorch must match. :contentReference[oaicite:2]{index=2}
# --------------------------------------------------------------------
RUN uv pip install --no-cache-dir \
      vllm \
      unsloth unsloth_zoo \
      trl datasets transformers gguf sentencepiece mistral_common tf-keras \
      "httpx>=0.24.0" \
      "aiometer>=0.5.0" \
      "aiosqlite>=0.19.0" \
      "jmespath>=1.0.0" \
      "tenacity>=8.2.0" \
      "tqdm>=4.65.0" \
      "pandas>=2.0.0" \
      "openpyxl>=3.1.0" \
      "pyarrow>=12.0.0" \
      "py-mini-racer>=0.6.0"

WORKDIR /workspace
CMD ["bash", "-lc", "sleep infinity"]
