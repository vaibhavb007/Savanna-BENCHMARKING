#!/usr/bin/env python3
"""
Unified Vision-Language Benchmark Harness (CPU fallback compatible)

- Works on Jetson systems even without CUDA-enabled PyTorch
- Uses vLLM or transformers, whichever is available
- Collects latency + total power via tegrastats
- Logs results to CSV for batch sizes

Author: adapted for CPU fallback (Savanna Coffel Jetson environment)
"""

import os, csv, time, threading, subprocess, re
from PIL import Image
import torch

# --------------------------
# Device / CUDA Guard
# --------------------------
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
if USE_CUDA:
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ CUDA not available — running in CPU fallback mode.")

# --------------------------
# CONFIGURATION
# --------------------------
BATCH_SIZES = [1, 2]  # keep small for CPU
IMAGES_DIR = "Images_LR"
CSV_OUT = "vlm_vllm_bench_cpu.csv"
PROMPTS = [
    "Is it a rural or an urban area?",
    "Is there a water area?",
    "Is there a small water area?",
    "Is a building present?",
]
MODEL_MAP = {
    # official LLaVA v1.5 from Haotian Liu (public)
    
    # Qwen vision-language
    "qwen2.5vl:3b": "Qwen/Qwen2.5-VL-3B-Instruct",

    # Gemma and others
    "gemma3:4b": "google/gemma-3-4b-it",
    "gemma3n:e2b": "google/gemma-3-4b-it",
}

# --------------------------
# Power Sampling Thread
# --------------------------
POWER_RE = re.compile(r"VDD_IN\s+(\d+)mW")
def sample_power(samples, stop_event, interval_ms=200):
    cmd = ["sudo", "tegrastats", "--interval", str(interval_ms)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        for line in proc.stdout:
            if stop_event.is_set():
                break
            m = POWER_RE.search(line)
            if m:
                samples.append((time.time(), float(m.group(1)) / 1000.0))
    finally:
        stop_event.set()
        proc.terminate()

def integrate_energy(samples):
    if len(samples) < 2:
        return 0.0, 0.0
    E = 0.0
    for (t1, p1), (t2, p2) in zip(samples[:-1], samples[1:]):
        E += 0.5 * (p1 + p2) * (t2 - t1)
    dur = samples[-1][0] - samples[0][0]
    return (E / dur if dur > 0 else 0.0), E

# --------------------------
# vLLM / transformers backend
# --------------------------
def run_inference(model_id, prompts, images):
    try:
        from vllm import LLM
        print(f"[vLLM] Loading model {model_id} (device={DEVICE})")
        llm = LLM(model=model_id, device=DEVICE)
        results = []
        for p, img in zip(prompts, images):
            out = llm.generate(p, multi_modal_data={"image": img})
            results.append(out[0].outputs[0].text.strip())
        return results

    except Exception as e:
        print(f"[vLLM failed or not installed] Falling back to transformers: {e}")
        from transformers import (
            LlavaProcessor,
            LlavaForConditionalGeneration,
            AutoProcessor,
            AutoModelForVision2Seq
        )

        # LLaVA models use special processor/model classes
        if "llava" in model_id.lower():
            print(f"[Transformers] Using LLaVA processor for {model_id}")
            processor = LlavaProcessor.from_pretrained(model_id)
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=(torch.float16 if USE_CUDA else torch.float32),
                device_map=DEVICE
            ).eval()
            inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=20)
            return processor.batch_decode(out, skip_special_tokens=True)

        # other VLMs (Qwen, Gemma, etc.)
        print(f"[Transformers] Using AutoProcessor for {model_id}")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=(torch.float16 if USE_CUDA else torch.float32),
            device_map=DEVICE,
            trust_remote_code=True
        ).eval()
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=20)
        return processor.batch_decode(out, skip_special_tokens=True)

# --------------------------
# Benchmark loop
# --------------------------
def benchmark_model(ollama_name):
    model_id = MODEL_MAP.get(ollama_name)
    if not model_id:
        print(f"❌ Unknown model {ollama_name}")
        return

    image_files = sorted([os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) if f.endswith(".tif")])
    all_images = [Image.open(f).convert("RGB") for f in image_files]

    with open(CSV_OUT, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["model", "batch_size", "latency_s", "avg_power_w", "energy_j"])

        for bs in BATCH_SIZES:
            images = all_images[:bs]
            prompts = PROMPTS[:bs]
            print(f"\n=== {ollama_name} | Batch {bs} ===")

            power_samples = []
            stop_event = threading.Event()
            sampler = threading.Thread(target=sample_power, args=(power_samples, stop_event, 200))
            sampler.start()

            if USE_CUDA:
                torch.cuda.synchronize()
            start = time.time()
            try:
                outputs = run_inference(model_id, prompts, images)
            finally:
                if USE_CUDA:
                    torch.cuda.synchronize()
                end = time.time()
                stop_event.set()
                sampler.join(timeout=2)

            dur = end - start
            avg_p, tot_e = integrate_energy(power_samples)
            for p, o in zip(prompts, outputs):
                print(f"{p} → {o}")
            print(f"⏱️ {dur:.2f}s | ⚡ {avg_p:.2f} W | 🔋 {tot_e:.2f} J")

            writer.writerow([ollama_name, bs, f"{dur:.2f}", f"{avg_p:.2f}", f"{tot_e:.2f}"])
            f.flush()

# --------------------------
# Entry
# --------------------------
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for m in [
        "qwen2.5vl:3b",
        "gemma3:4b",
        "gemma3n:e2b",
    ]:
        benchmark_model(m)

