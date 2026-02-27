FROM runpod/base:0.6.2-cuda12.4.1
# System deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
# Python deps
RUN python3.11 -m pip install --no-cache-dir \
    vllm==0.7.3 \
    runpod \
    transformers==4.57 \
    accelerate \
    huggingface_hub \
    hf_transfer \
    openai \
    httpx \
    Pillow
# Download model at BUILD time
ARG HF_TOKEN
RUN python3.11 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='/model', token='${HF_TOKEN}')"
# Patch conflicting rope_scaling config
RUN python3.11 -c "import json; path='/model/config.json'; cfg=json.load(open(path)); rope=cfg.get('text_config',{}).get('rope_scaling',{}); rope.pop('type',None); json.dump(cfg,open(path,'w'),indent=2)"
# Copy handler
COPY handler.py /handler.py
CMD ["python3.11", "-u", "/handler.py"]
