import pandas as pd
import os

# === Base directory for all benchmark CSVs ===
base_path = "/Users/savannacoffel/Desktop/Savanna-BENCHMARKING/power benchmarking"

# === Model → CSV mapping for each benchmark type ===
equality_files = {
    "Gemma-3.1B": "TEGRA/benchmark_results_EQUALITYgemma3_1bTEGRA.csv",
    "Gemma-3n_e2b": "TEGRA/benchmark_results_EQUALITYgemma3n_e2bTEGRA.csv",
    "LLaVA-7B": "TEGRA/benchmark_results_EQUALITYllava_7bTEGRA.csv",
    "LLaVA-Llama3": "TEGRA/benchmark_results_EQUALITYllava-llama3_latestTEGRA.csv",
    "Qwen2.5VL-3B": "TEGRA/benchmark_results_EQUALITYqwen2.5vl_3bTEGRA.csv",
    "Qwen2.5VL-7B": "TEGRA/benchmark_results_EQUALITYqwen2.5vl_7bTEGRA.csv",
}

yesno_files = {
    "Gemma-3.1B": "TEGRA/benchmark_results_YESNOgemma3_1bTEGRA.csv",
    "Gemma-3n_e2b": "TEGRA/benchmark_results_YESNOgemma3n_e2bTEGRA.csv",
    "LLaVA-7B": "TEGRA/benchmark_results_YESNOllava_7bTEGRA.csv",
    "LLaVA-Llama3": "TEGRA/benchmark_results_YESNOllava-llama3_latestTEGRA.csv",
    "Qwen2.5VL-3B": "TEGRA/benchmark_results_YESNOqwen2.5vl_3bTEGRA.csv",
    "Qwen2.5VL-7B": "TEGRA/benchmark_results_YESNOqwen2.5vl_7bTEGRA.csv",
}

# === Function to compute benchmark summary ===
def compute_summary(files_dict, base_path):
    summary = []

    for model, filename in files_dict.items():
        file_path = os.path.join(base_path, filename)
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)

        # Ensure numeric and clean data
        df["latency_sec"] = pd.to_numeric(df["latency_sec"], errors="coerce")
        df["avg_tot_w"] = pd.to_numeric(df["avg_tot_w"], errors="coerce")
        df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
        df = df.dropna(subset=["latency_sec", "avg_tot_w", "correct"])

        # Compute metrics
        avg_latency = df["latency_sec"].mean()
        med_power = df["avg_tot_w"].median()
        avg_power = df["avg_tot_w"].mean()
        accuracy = (df["correct"] == 1).mean() * 100
        total_energy = (df["avg_tot_w"] * df["latency_sec"]).sum()

        summary.append({
            "Model": model,
            "Avg Latency (s)": avg_latency,
            "Median Power of avg_tot_w (W) per query": med_power,
            "Avg Power of avg_tot_w (W) per query": avg_power,
            "Accuracy (%)": accuracy,
            "Total Energy (J) over all queries (∑(avg_tot_w×time))": total_energy
        })

    summary_df = pd.DataFrame(summary).round(2)
    return summary_df


# === Compute summaries for both datasets ===
equality_summary = compute_summary(equality_files, base_path)
yesno_summary = compute_summary(yesno_files, base_path)

# === Print summaries to console ===
print("\n=== EQUALITY BENCHMARK SUMMARY WITH TEGRASTATS ===")
print(equality_summary.to_string(index=False))

print("\n=== YES/NO BENCHMARK SUMMARY WITH TEGRASTATS ===")
print(yesno_summary.to_string(index=False))

# === Save both tables to CSV ===
equality_path = os.path.join(base_path, "benchmark_summary_EQUALITYTEGRA.csv")
yesno_path = os.path.join(base_path, "benchmark_summary_YESNOTEGRA.csv")

equality_summary.to_csv(equality_path, index=False)
yesno_summary.to_csv(yesno_path, index=False)

# === Save Markdown tables for display in docs or notebooks ===
equality_md_path = os.path.join(base_path, "benchmark_summary_EQUALITYTEGRA.md")
yesno_md_path = os.path.join(base_path, "benchmark_summary_YESNOTEGRA.md")

# equality_summary.to_markdown(equality_md_path, index=False)
# yesno_summary.to_markdown(yesno_md_path, index=False)


print(f" EQUALITY summary saved to: {equality_path}")
print(f" YES/NO summary saved to: {yesno_path}")
print(f" Markdown tables saved to:\n  {equality_md_path}\n  {yesno_md_path}")