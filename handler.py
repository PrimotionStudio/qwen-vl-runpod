import runpod
from vllm import LLM, SamplingParams

# This runs ONCE when the container starts
llm = LLM(
    model="/model",
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.9,
    enforce_eager=True  # faster cold start
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=512
)

def handler(job):
    prompt = job["input"].get("prompt", "Hello")

    outputs = llm.generate(prompt, sampling_params)
    text = outputs[0].outputs[0].text

    return {"output": text}

runpod.serverless.start({"handler": handler})
