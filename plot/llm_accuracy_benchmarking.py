#!/usr/bin/env python3
import json
import time
import os
import string
import csv
import argparse
import subprocess
import threading
import re
import statistics

# --------------------------
# Utility functions
# --------------------------
def normalize(text):
    return text.strip().lower().translate(str.maketrans("", "", string.punctuation))

def run_ollama_cli(model, prompt, image_path):
    """Run Ollama CLI for one multimodal query and return output + latency."""
    start = time.time()
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt, image_path],
            capture_output=True,
            text=True,
            check=True
        )
        latency = time.time() - start
        output = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        enforced = output.strip().split()[0].lower() if output else ""
        return enforced, latency
    except subprocess.CalledProcessError as e:
        print(f"Error running ollama: {e.stderr}", flush=True)
        return "", 0.0

# --------------------------
# tegrastats power sampler
# --------------------------
POWER_TOTAL_RE = re.compile(r"VDD_IN\s+(\d+)mW")
POWER_CPU_RE   = re.compile(r"VDD_CPU_GPU_CV\s+(\d+)mW")
POWER_SOC_RE   = re.compile(r"VDD_SOC\s+(\d+)mW")

def sample_power_tegrastats(samples, stop_event, interval_ms=50):
    """
    Collect total, CPU/GPU, and SOC power samples from tegrastats at the specified interval (ms).
    Stores dicts with timestamp and power in W.
    """
    cmd = ["sudo", "tegrastats", "--interval", str(interval_ms)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, bufsize=1)
    try:
        for line in proc.stdout:
            if stop_event.is_set():
                break
            m_tot = POWER_TOTAL_RE.search(line)
            m_cpu = POWER_CPU_RE.search(line)
            m_soc = POWER_SOC_RE.search(line)
            if m_tot:
                samples.append({
                    "t": time.time(),
                    "tot": float(m_tot.group(1)) / 1000.0,
                    "cpu_gpu": float(m_cpu.group(1)) / 1000.0 if m_cpu else 0.0,
                    "soc": float(m_soc.group(1)) / 1000.0 if m_soc else 0.0
                })
    except Exception as e:
        print(f"[tegrastats] Sampling error: {e}", flush=True)
    finally:
        stop_event.set()
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()

# --------------------------
# Energy integration
# --------------------------
def integrate_energy(samples):
    """Return trapezoidal average power (W) and total energy (J)."""
    if len(samples) < 2:
        return 0.0, 0.0
    E = 0.0
    for a, b in zip(samples[:-1], samples[1:]):
        dt = b["t"] - a["t"]
        E += 0.5 * (a["tot"] + b["tot"]) * dt
    duration = samples[-1]["t"] - samples[0]["t"]
    avg_power = E / duration if duration > 0 else 0.0
    return avg_power, E

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=None)
    parser.add_argument("--answers", default=None)
    parser.add_argument("--image-dir", default="Images_LR")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--index", type=int, required=True)
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

    # ---- Run inference + measure power ----
    power_samples = []
    stop_event = threading.Event()
    sampler = threading.Thread(target=sample_power_tegrastats,
                               args=(power_samples, stop_event, 50))  # 50ms interval
    sampler.start()

    start_time = time.time()
    response, latency = run_ollama_cli(args.model, question_text, image_path)
    end_time = time.time()

    stop_event.set()
    sampler.join(timeout=2)

    # ---- Sampling interval stats ----
    if len(power_samples) > 1:
        intervals = [(b["t"] - a["t"]) * 1000 for a, b in zip(power_samples[:-1], power_samples[1:])]
        print(f"Collected {len(power_samples)} samples")
        print(f"Average interval: {sum(intervals)/len(intervals):.1f} ms (min {min(intervals):.1f}, max {max(intervals):.1f})")
    else:
        print("No samples collected!")

    # ---- Keep only samples inside inference window ----
    samples_in_window = [s for s in power_samples if start_time <= s["t"] <= end_time]

    # ---- Compute energy and stats ----
    if samples_in_window:
        avg_tot = sum(s["tot"] for s in samples_in_window) / len(samples_in_window)
        max_tot = max(s["tot"] for s in samples_in_window)
        avg_cpu_gpu = sum(s["cpu_gpu"] for s in samples_in_window) / len(samples_in_window)
        max_cpu_gpu = max(s["cpu_gpu"] for s in samples_in_window)
        avg_soc = sum(s["soc"] for s in samples_in_window) / len(samples_in_window)
        max_soc = max(s["soc"] for s in samples_in_window)
        avg_power_integrated_w, total_energy_j = integrate_energy(samples_in_window)
    else:
        avg_tot = max_tot = avg_cpu_gpu = max_cpu_gpu = avg_soc = max_soc = avg_power_integrated_w = total_energy_j = 0.0

    is_correct = normalize(response) in gt_answers

    # ---- Write CSV ----
    file_exists = os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "question_id", "latency_sec", "correct",
                "model_response", "ground_truth", "question_text",
                "avg_tot_w", "max_tot_w",
                "avg_cpu_gpu_w", "max_cpu_gpu_w",
                "avg_soc_w", "max_soc_w",
                "avg_power_integrated_w", "total_energy_j"
            ])
        writer.writerow([
            qid, f"{latency:.3f}", int(is_correct),
            response, "|".join(gt_answers), q["question"],
            f"{avg_tot:.2f}", f"{max_tot:.2f}",
            f"{avg_cpu_gpu:.2f}", f"{max_cpu_gpu:.2f}",
            f"{avg_soc:.2f}", f"{max_soc:.2f}",
            f"{avg_power_integrated_w:.2f}", f"{total_energy_j:.2f}"
        ])
        f.flush()

    # ---- Print results ----
    print(f"[Q{qid}] {q['question']}", flush=True)
    print(f"Model: {response}", flush=True)
    print(f"GT: {gt_answers}", flush=True)
    print(f"Correct: {is_correct}, Time: {latency:.2f}s", flush=True)
    print(f"Power: tot avg {avg_tot:.2f} W, max {max_tot:.2f} W", flush=True)
    print(f"CPU+GPU: avg {avg_cpu_gpu:.2f} W, max {max_cpu_gpu:.2f} W", flush=True)
    print(f"SOC: avg {avg_soc:.2f} W, max {max_soc:.2f} W", flush=True)
    print(f"Integrated (avg) power: {avg_power_integrated_w:.2f} W, energy: {total_energy_j:.2f} J", flush=True)

if __name__ == "__main__":
    main()

