FROM runpod/base:0.6.2-cuda12.1.0

# System deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip3 install --no-cache-dir \
    vllm==0.6.3 \
    runpod \
    transformers \
    accelerate \
    huggingface_hub \
    Pillow

# Download model at BUILD time
ARG HF_TOKEN
RUN pip3 install huggingface_hub && \
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='/model', token='${HF_TOKEN}')"

# Copy handler
COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
