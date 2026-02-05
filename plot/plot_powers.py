import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to your folder
base_path = os.path.expanduser("~/Documents/Savanna-BENCHMARKING/")
save_dir = os.path.expanduser("~/Documents/Savanna-BENCHMARKING/graphs/")

# File names
files = {
    "Gemma3-1B": "benchmark_results_YESNOgemma3_1bTEGRA.csv",
    "Gemma3n-E2B": "benchmark_results_YESNOgemma3n_e2bTEGRA.csv", 
    "LLaVA-7B": "benchmark_results_YESNOllava_7bTEGRA.csv",
    "LLaVA-Llama3 Latest": "benchmark_results_YESNOllava-llama3_latestTEGRA.csv",
    "Qwen2.5VL-3B": "benchmark_results_YESNOqwen2.5vl_3bTEGRA.csv",
    "Qwen2.5VL-7B": "benchmark_results_YESNOqwen2.5vl_7bTEGRA.csv",
    "Gemma3:4B": "benchmark_results_YESNOgemma3_4b.csv"
}

# Columns we care about
power_columns = ["avg_tot_w", "max_tot_w", "avg_power_integrated_w"]

plt.style.use("seaborn-darkgrid")

for col in power_columns:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (label, filename) in zip(axes, files.items()):
        file_path = os.path.join(base_path, filename)
        df = pd.read_csv(file_path)

        if col not in df.columns:
            raise ValueError(f"File {filename} does not contain '{col}' column.")

        iterations = range(1, len(df) + 1)
        smoothed = df[col].rolling(window=10, min_periods=1).mean()

        ax.plot(iterations, smoothed, linewidth=2.5, color="tab:blue")

        if col == "avg_tot_w":
            median_val = df[col].median()
            ax.axhline(y=median_val, color="gray", linestyle="--", alpha=0.5)  # optional visual reference
            ax.legend([f"Median = {median_val:.2f} W"], loc="upper right", frameon=True)

        ax.set_title(label, fontsize=12, weight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)

    # Add shared labels
    fig.suptitle(f"Benchmark Comparison for Questions of Type 'YES/NO' Using TegraStats: {col}", fontsize=16, weight="bold")
    fig.text(0.5, 0.04, "Iteration", ha="center", fontsize=14)
    fig.text(0.04, 0.5, "Power (Watts)", va="center", rotation="vertical", fontsize=14)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    save_path = os.path.join(save_dir, f"benchmark_{col}_subplots.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")

    plt.show()

