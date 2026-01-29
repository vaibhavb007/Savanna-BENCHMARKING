import json
import time
import os
import string
import csv
import argparse
import subprocess
import threading
from jtop import jtop  # Jetson stats

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

def sample_power(jetson, samples, stop_event, interval=0.1):
    """Background sampler."""
    while not stop_event.is_set() and jetson.ok():
        p = jetson.power
        if p:
            samples.append({
                "t": time.time(),
                "tot": p["tot"]["power"] / 1000.0,
                "cpu_gpu": p["rail"]["VDD_CPU_GPU_CV"]["power"] / 1000.0,
                "soc": p["rail"]["VDD_SOC"]["power"] / 1000.0
            })
        time.sleep(interval)

def integrate_energy(samples):
    """Compute average power in watts"""
    E = 0.0
    for a, b in zip(samples[:-1], samples[1:]):
        dt = b["t"] - a["t"]
        E += 0.5 * (a["tot"] + b["tot"]) * dt
        duration = samples[-1]["t"] - samples[0]["t"] 
    return E / duration if duration > 0 else 0.0

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

    # ---- Run inference + measure per-call power/energy ----
    power_samples = []
    with jtop(interval=0.1) as jetson:  # 100 ms polling
        # Wait until jtop starts returning valid data
        print("Waiting for jtop to start streaming power data...", flush=True)
        t0 = time.time()
        while not jetson.ok() or not jetson.power:
            time.sleep(0.1) # wait
            if time.time() - t0 > 5:  # timeout after 5 seconds
                print("jtop failed to initialize after 5s", flush=True)
                return

            print(" jtop ready, starting inference measurement...", flush=True)

        stop_event = threading.Event()
        sampler = threading.Thread(target=sample_power, args=(jetson, power_samples, stop_event))
        sampler.start()

        start_time = time.time()
        response, latency = run_ollama_cli(args.model, question_text, image_path)
        end_time = time.time()

        stop_event.set()
        sampler.join(timeout=2)

    # ---- Verify actual sampling interval ----
    if len(power_samples) > 1:
        intervals = [(b["t"] - a["t"]) * 1000 for a, b in zip(power_samples[:-1], power_samples[1:])]
        print(f"Collected {len(power_samples)} samples")
        print(f"Average interval: {sum(intervals)/len(intervals):.1f} ms")
        print(f"Min: {min(intervals):.1f} ms, Max: {max(intervals):.1f} ms")
    else:
        print("No samples collected!")

    # Keep only samples inside the inference window
    samples_in_window = [s for s in power_samples if start_time <= s["t"] <= end_time]

    # Compute stats and energy
    if samples_in_window:
        avg_tot = sum(s["tot"] for s in samples_in_window) / len(samples_in_window)
        max_tot = max(s["tot"] for s in samples_in_window)
        avg_cpu_gpu = sum(s["cpu_gpu"] for s in samples_in_window) / len(samples_in_window)
        max_cpu_gpu = max(s["cpu_gpu"] for s in samples_in_window)
        avg_power_integrated_w = integrate_energy(samples_in_window)
    else:
        avg_tot = max_tot = avg_cpu_gpu = max_cpu_gpu = avg_power_integrated_w = 0.0

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
                "avg_power_integrated_w"
            ])
        writer.writerow([
            qid, f"{latency:.3f}", int(is_correct),
            response, "|".join(gt_answers), q["question"],
            f"{avg_tot:.2f}", f"{max_tot:.2f}",
            f"{avg_cpu_gpu:.2f}", f"{max_cpu_gpu:.2f}",
            f"{avg_power_integrated_w:.2f}"
        ])
        f.flush()

    # ---- Print results ----
    print(f"[Q{qid}] {q['question']}", flush=True)
    print(f"Model: {response}", flush=True)
    print(f"GT: {gt_answers}", flush=True)
    print(f"Correct: {is_correct}, Time: {latency:.2f}s", flush=True)
    print(f"Power: tot avg {avg_tot:.2f} W, tot max {max_tot:.2f} W, "
          f"cpu+gpu avg {avg_cpu_gpu:.2f} W, cpu+gpu max {max_cpu_gpu:.2f} W", flush=True)
    print(f"Energy: {avg_power_integrated_w:.2f} W", flush=True)

if __name__ == "__main__":
    main()

