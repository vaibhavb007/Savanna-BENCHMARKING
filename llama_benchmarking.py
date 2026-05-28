import json
import time
import os
import string
import csv
import argparse
import subprocess
import threading
import re
import base64
from openai import OpenAI

# ---------------- Utilities ---------------- #

def normalize(text):
    return text.strip().lower().translate(str.maketrans("", "", string.punctuation))

# ---------------- API Inference (llama.cpp) ------------------ #

def run_llamacpp_serve(model, prompt, image_path):
    start = time.time()
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                        },
                    ],
                }
            ],
            max_completion_tokens=128,
            temperature=0.0
        )

        latency = time.time() - start

        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip(), latency
        return "No response from model", latency

    except Exception as e:
        print(f"Error running llama.cpp API: {e}", flush=True)
        return "", 0.0


# ---------------- Tegrastats Power Sampling ---------------- #

# Matches: VDD_CPU_GPU_CV 402mW/402mW  →  group(1) = current draw in mW
_CPU_GPU_CV_RE = re.compile(r"VDD_CPU_GPU_CV\s+(\d+)mW/\d+mW")

# Matches: VDD_IN 4790mW/4790mW  →  group(1) = current total board draw in mW
_VDD_IN_RE = re.compile(r"VDD_IN\s+(\d+)mW/\d+mW")

# Matches: VDD_SOC 1569mW/1569mW  →  group(1) = SoC fabric rail in mW
# This is stable idle overhead; VDD_IN is the natural upper bound across all rails.
_VDD_SOC_RE = re.compile(r"VDD_SOC\s+(\d+)mW/\d+mW")

# Matches: gpu@48.781C  →  group(1) = GPU die temp in Celsius
_GPU_TEMP_RE = re.compile(r"gpu@([\d.]+)C")


class TegrastatsSampler:
    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.process = None
        self.current_cpu_gpu_cv_w = 0.0  # VDD_CPU_GPU_CV in watts
        self.current_vdd_in_w = 0.0      # VDD_IN (total board) in watts
        self.current_vdd_soc_w = 0.0     # VDD_SOC (SoC fabric, stable idle baseline) in watts
        self.current_gpu_temp_c = 0.0
        self.running = False

    def start(self):
        self.running = True
        self.process = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.thread = threading.Thread(target=self._parse_output, daemon=True)
        self.thread.start()

    def _parse_output(self):
        while self.running and self.process and self.process.poll() is None:
            line = self.process.stdout.readline()
            if not line:
                continue

            m = _CPU_GPU_CV_RE.search(line)
            if m:
                self.current_cpu_gpu_cv_w = float(m.group(1)) / 1000.0

            m = _VDD_IN_RE.search(line)
            if m:
                self.current_vdd_in_w = float(m.group(1)) / 1000.0

            m = _VDD_SOC_RE.search(line)
            if m:
                self.current_vdd_soc_w = float(m.group(1)) / 1000.0

            m = _GPU_TEMP_RE.search(line)
            if m:
                self.current_gpu_temp_c = float(m.group(1))

    def shutdown(self):
        self.running = False
        if self.process:
            self.process.terminate()
            self.process.wait()


def sample_power_tegrastats(sampler, samples, stop_event, interval=0.1):
    """Background thread: snapshot the latest parsed tegrastats values."""
    while not stop_event.is_set():
        samples.append({
            "t": time.time(),
            "cpu_gpu_cv_w": sampler.current_cpu_gpu_cv_w,
            "vdd_in_w":     sampler.current_vdd_in_w,
            "vdd_soc_w":    sampler.current_vdd_soc_w,
            "temp":         sampler.current_gpu_temp_c,
        })
        time.sleep(interval)


def integrate_energy(samples, key):
    """
    Compute time-averaged power (watts) for the given sample key
    via trapezoidal integration over the sample window.
    """
    if len(samples) < 2:
        return samples[0][key] if samples else 0.0

    E = 0.0
    for a, b in zip(samples[:-1], samples[1:]):
        dt = b["t"] - a["t"]
        E += 0.5 * (a[key] + b[key]) * dt

    duration = samples[-1]["t"] - samples[0]["t"]
    return E / duration if duration > 0 else 0.0


# ---------------- Main ---------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--image-dir", default="Images_LR")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--engine", required=True, choices=["llamacpp"])
    args = parser.parse_args()

    if args.output is None:
        safe_model = args.model.replace(":", "_").replace("/", "_")
        args.output = f"benchmark_results_{safe_model}.csv"

    with open(args.questions) as f:
        all_questions = json.load(f)["questions"]

    with open(args.answers) as f:
        all_answers = {a["id"]: a for a in json.load(f)["answers"]}

    if args.index >= len(all_questions):
        print(f"❌ Index {args.index} out of range", flush=True)
        return

    q = all_questions[args.index]
    qid = q["id"]
    img_id = q["img_id"]
    image_path = os.path.join(args.image_dir, f"{img_id}.tif")

    if not os.path.exists(image_path):
        print(f"Missing image {image_path}, skipping.", flush=True)
        return

    question_text = q["question"] + "\nAnswer with exactly one word or number only. Do not explain."
    gt_answers = [
        normalize(all_answers[aid]["answer"])
        for aid in q.get("answers_ids", [])
        if aid in all_answers
    ]

    if not gt_answers:
        print(f"No ground truth for qid {qid}, skipping.", flush=True)
        return

    # -------- Tegrastats Power Measurement -------- #

    power_samples = []
    sampler = TegrastatsSampler(interval_ms=100)
    sampler.start()

    # Brief warm-up: give tegrastats time to emit its first line before
    # inference starts, so the very first sample isn't a stale zero.
    time.sleep(0.25)

    stop_event = threading.Event()
    sampler_thread = threading.Thread(
        target=sample_power_tegrastats,
        args=(sampler, power_samples, stop_event),
        daemon=True
    )

    sampler_thread.start()
    start_time = time.time()

    response, latency = run_llamacpp_serve(args.model, question_text, image_path)

    end_time = time.time()
    stop_event.set()
    sampler_thread.join(timeout=2)
    sampler.shutdown()

    # -------- Stats -------- #

    samples_in_window = [s for s in power_samples if start_time <= s["t"] <= end_time]

    if samples_in_window:
        avg_cpu_gpu_cv_w  = sum(s["cpu_gpu_cv_w"] for s in samples_in_window) / len(samples_in_window)
        max_cpu_gpu_cv_w  = max(s["cpu_gpu_cv_w"] for s in samples_in_window)
        avg_vdd_in_w      = sum(s["vdd_in_w"]     for s in samples_in_window) / len(samples_in_window)
        max_vdd_in_w      = max(s["vdd_in_w"]     for s in samples_in_window)
        avg_vdd_soc_w     = sum(s["vdd_soc_w"]    for s in samples_in_window) / len(samples_in_window)
        max_vdd_soc_w     = max(s["vdd_soc_w"]    for s in samples_in_window)
        max_gpu_temp_c    = max(s["temp"]          for s in samples_in_window)
        int_cpu_gpu_cv_w  = integrate_energy(samples_in_window, "cpu_gpu_cv_w")
        int_vdd_in_w      = integrate_energy(samples_in_window, "vdd_in_w")
        int_vdd_soc_w     = integrate_energy(samples_in_window, "vdd_soc_w")
    else:
        avg_cpu_gpu_cv_w = max_cpu_gpu_cv_w = 0.0
        avg_vdd_in_w     = max_vdd_in_w     = 0.0
        avg_vdd_soc_w    = max_vdd_soc_w    = 0.0
        max_gpu_temp_c                       = 0.0
        int_cpu_gpu_cv_w = int_vdd_in_w = int_vdd_soc_w = 0.0

    is_correct = normalize(response) in gt_answers

    # -------- CSV Output -------- #

    file_exists = os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "question_id", "latency_sec", "correct",
                "model_response", "ground_truth", "question_text",
                # VDD_CPU_GPU_CV  (CPU + GPU + CV rail — closest proxy for inference load)
                "avg_cpu_gpu_cv_w", "max_cpu_gpu_cv_w", "integrated_cpu_gpu_cv_w",
                # VDD_SOC  (SoC fabric idle baseline — stable overhead reference)
                "avg_vdd_soc_w", "max_vdd_soc_w", "integrated_vdd_soc_w",
                # VDD_IN   (total board power — upper bound, sum of all rails)
                "avg_vdd_in_w", "max_vdd_in_w", "integrated_vdd_in_w",
                "max_gpu_temp_c",
            ])
        writer.writerow([
            qid, f"{latency:.3f}", int(is_correct),
            response, "|".join(gt_answers), q["question"],
            f"{avg_cpu_gpu_cv_w:.2f}",  f"{max_cpu_gpu_cv_w:.2f}",  f"{int_cpu_gpu_cv_w:.2f}",
            f"{avg_vdd_soc_w:.2f}",     f"{max_vdd_soc_w:.2f}",     f"{int_vdd_soc_w:.2f}",
            f"{avg_vdd_in_w:.2f}",      f"{max_vdd_in_w:.2f}",      f"{int_vdd_in_w:.2f}",
            f"{max_gpu_temp_c:.1f}",
        ])

    # -------- Console -------- #

    print(f"[Q{qid}]", flush=True)
    print(f"Engine:   {args.engine}", flush=True)
    print(f"Response: {response}", flush=True)
    print(f"GT:       {gt_answers}", flush=True)
    print(f"Correct:  {is_correct},  Time: {latency:.2f}s", flush=True)
    print(f"VDD_CPU_GPU_CV:  avg {avg_cpu_gpu_cv_w:.2f} W,  max {max_cpu_gpu_cv_w:.2f} W  (integrated {int_cpu_gpu_cv_w:.2f} W)", flush=True)
    print(f"VDD_SOC:         avg {avg_vdd_soc_w:.2f} W,  max {max_vdd_soc_w:.2f} W  (integrated {int_vdd_soc_w:.2f} W)", flush=True)
    print(f"VDD_IN (board):  avg {avg_vdd_in_w:.2f} W,  max {max_vdd_in_w:.2f} W  (integrated {int_vdd_in_w:.2f} W)", flush=True)
    print(f"GPU temp (max):  {max_gpu_temp_c:.1f} C", flush=True)


if __name__ == "__main__":
    main()
