FROM python:3.10-slim-bookworm

# Install build dependencies (including wget for install_libjpegturbo.sh)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config git curl wget unzip \
    libjpeg-dev libpng-dev libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and run install scripts
# (module load commands fail silently with || true)
COPY install_libjpegturbo.sh install_opencv.sh install_aws_cli.sh /tmp/

RUN chmod +x /tmp/install_*.sh && \
    /tmp/install_libjpegturbo.sh && \
    /tmp/install_opencv.sh && \
    /tmp/install_aws_cli.sh && \
    rm /tmp/install_*.sh

# Update library cache and set environment
RUN ldconfig
ENV PATH="/root/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="/root/.local/lib:$LD_LIBRARY_PATH"
ENV PKG_CONFIG_PATH="/root/.local/lib/pkgconfig:$PKG_CONFIG_PATH"

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install s5cmd
RUN curl -L "https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz" -o s5cmd.tar.gz && \
    tar -xzf s5cmd.tar.gz && \
    mv s5cmd /usr/local/bin/ && \
    chmod +x /usr/local/bin/s5cmd && \
    rm s5cmd.tar.gz

# Install JupyterLab
RUN pip install --no-cache-dir jupyterlab ipykernel
COPY install_uv_kernels.sh /tmp/

RUN chmod +x /tmp/install_*.sh && \
    /tmp/install_uv_kernels.sh && \
    rm /tmp/install_*.sh

WORKDIR /workspace
