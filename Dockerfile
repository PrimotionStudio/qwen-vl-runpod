FROM runpod/base:0.6.2-cuda12.1.0

# Install Python libraries
RUN pip install --no-cache-dir \
    vllm \
    transformers \
    accelerate \
    huggingface_hub

# Download the model DURING build (this is the key optimization)
RUN python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
    local_dir="/model",
    local_dir_use_symlinks=False
)
EOF

# Copy your inference code
COPY handler.py /handler.py

# Start the server
CMD ["python", "-u", "/handler.py"]
