import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to your folder
base_path = os.path.expanduser("~/Documents/Savanna-BENCHMARKING/")
save_path = os.path.expanduser("~/Documents/Savanna-BENCHMARKING/benchmark_accuracy_table.png")

# File names
files = {
    "Gemma3-4B": "benchmark_results_gemma3_4b.csv",
    "LLaVA-7B": "benchmark_results_llava_7b.csv",
    "LLaVA-Llama3 Latest": "benchmark_results_llava-llama3_latest.csv",
    "Qwen2.5VL-3B": "benchmark_results_qwen2.5vl:3b.csv"
}

# Collect accuracies
accuracies = {}
for label, filename in files.items():
    file_path = os.path.join(base_path, filename)
    df = pd.read_csv(file_path)
    if "correct" not in df.columns:
        raise ValueError(f"File {filename} does not contain 'correct' column.")
    accuracies[label] = df["correct"].mean() * 100  # percentage

# Convert to DataFrame
acc_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])
acc_df = acc_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
acc_df["Accuracy (%)"] = acc_df["Accuracy"].map(lambda x: f"{x:.2f}%")
acc_df = acc_df.drop(columns=["Accuracy"])  # keep only formatted column

# Plot table
fig, ax = plt.subplots(figsize=(6, 3))
ax.axis("off")  # Hide axes

# Add title
plt.title("Accuracy Collected Over 1000 Runs - Low Res Dataset",
          fontsize=14, weight="bold", pad=20)

# Create table
table = ax.table(
    cellText=acc_df.values,
    colLabels=acc_df.columns,
    cellLoc="center",
    loc="center"
)

# Style table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Accuracy table saved to {save_path}")

plt.show()

