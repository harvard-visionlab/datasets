FROM python:3.10-slim-bookworm

# Install system dependencies
# libjpeg-turbo8 is used automatically by PIL for faster JPEG decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget unzip \
    libjpeg-turbo-progs libjpeg62-turbo-dev \
    libpng-dev libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI
COPY install_aws_cli.sh /tmp/
RUN chmod +x /tmp/install_aws_cli.sh && \
    /tmp/install_aws_cli.sh && \
    rm /tmp/install_aws_cli.sh

ENV PATH="/root/.local/bin:$PATH"

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
