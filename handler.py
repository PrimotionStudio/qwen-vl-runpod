import runpod
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
import base64
from io import BytesIO

# Loaded ONCE on cold start — not per request
llm = LLM(
    model="/model",
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    limit_mm_per_prompt={"image": 5},
)

def handler(job):
    job_input = job["input"]

    # Optional keep-warm ping
    if job_input.get("ping"):
        return {"pong": True}

    prompt = job_input.get("prompt", "Describe this image.")
    image_b64 = job_input.get("image_base64")  # optional image

    messages = [{"role": "user", "content": []}]

    if image_b64:
        image_bytes = base64.b64decode(image_b64)
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
        })

    messages[0]["content"].append({"type": "text", "text": prompt})

    from vllm.entrypoints.chat_utils import apply_chat_template
    outputs = llm.chat(messages=messages, sampling_params=SamplingParams(
        temperature=job_input.get("temperature", 0.7),
        max_tokens=job_input.get("max_tokens", 512),
    ))

    return {"output": outputs[0].outputs[0].text}

runpod.serverless.start({"handler": handler})
