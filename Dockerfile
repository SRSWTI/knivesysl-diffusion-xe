# AXE-Diffusion KSL-XE runtime image.
#
# Base: local/gemma-w4-intel:tested (torch 2.12.0+xpu, Triton XPU stack).
# Upgrades Triton to 3.8.0 from the PyTorch XPU index and embeds the
# server, studio UI, and W4A16 kernels for standalone use.
# docker-compose.yml overlays the server, both UIs, and ./kernels
# read-only; restart the service to load Python changes without rebuilding.
#
# Build (also done by scripts/xe.sh up when the image is missing):
#   docker build --build-arg BASE_IMAGE=local/gemma-w4-intel:tested \
#     -t local/axe-diffusion-ksl-xe:triton38 .

ARG BASE_IMAGE=local/gemma-w4-intel:tested
FROM ${BASE_IMAGE}

# Triton 3.8.0 XPU backend — the version the W4A16 kernels target.
RUN pip install --no-cache-dir 'triton-xpu==3.8.0' \
    --index-url https://download.pytorch.org/whl/xpu

COPY server.py web.html editor.html /app/
COPY kernels/ /app/kernels/
WORKDIR /app

# Replace the base image's sleep/infinity entrypoint so the image runs
# the server standalone; compose overrides entrypoint/command anyway.
ENTRYPOINT ["python"]
CMD ["-u", "/app/server.py"]
