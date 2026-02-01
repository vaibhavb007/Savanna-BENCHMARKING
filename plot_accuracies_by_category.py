import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Path
base_path = os.path.expanduser("~/Documents/Savanna-BENCHMARKING/")
save_path = os.path.expanduser("~/Documents/Savanna-BENCHMARKING/benchmark_accuracy_by_qtype.png")

# File names
files = {
    "Gemma3-4B": "benchmark_results_gemma3_4b.csv",
    "LLaVA-7B": "benchmark_results_llava_7b.csv",
    "LLaVA-Llama3 Latest": "benchmark_results_llava-llama3_latest.csv",
    "Qwen2.5VL-3B": "benchmark_results_qwen2.5vl:3b.csv"
}

# Categorization function
def categorize_question(text):
    text = text.lower().strip()

    if "equal to" in text:
        return "Equality"
    elif "more" in text or "less" in text:
        return "Comparison"
    elif text.startswith(("what", "how")) and ("number" in text or "amount" in text or "many" in text):
        return "Counting"
    elif any(word in text for word in ["small", "large", "circular", "square", "rectangular"]):
        return "Attribute"
    elif " at the " in text:
        return "Location"
    elif text.startswith(("is", "are", "does", "do", "was")):
        return "Yes/No"
    else:
        return "Other"

# Collect results
results = []

for model, filename in files.items():
    file_path = os.path.join(base_path, filename)
    df = pd.read_csv(file_path)

    if "correct" not in df.columns or "question_text" not in df.columns:
        raise ValueError(f"File {filename} must contain 'correct' and 'question_text' columns.")

    df["qtype"] = df["question_text"].apply(categorize_question)

    grouped = df.groupby("qtype")["correct"].mean().reset_index()
    grouped["Accuracy"] = grouped["correct"] * 100
    grouped["Model"] = model

    results.append(grouped[["Model", "qtype", "Accuracy"]])

results_df = pd.concat(results, ignore_index=True)

# Plot
plt.figure(figsize=(12, 7))  # wider figure
sns.barplot(
    data=results_df, 
    x="qtype", y="Accuracy", hue="Model", 
    palette="Set2", width=0.7  # narrower bars so spacing is clearer
)

# Title
plt.title("Model Accuracy by Question Type (percent correct within each category)",
          fontsize=14, weight="bold")

# Axis labels
plt.ylabel("Accuracy (%)", fontsize=12)
plt.xlabel("Question Type", fontsize=12)
plt.ylim(0, 100)

# Rotate x labels for clarity
plt.xticks(rotation=15)

# Add labels above bars (nudge more upwards)
for p in plt.gca().patches:
    height = p.get_height()
    if height > 0:
        plt.gca().text(
            p.get_x() + p.get_width() / 2.,
            height + 2,   # push labels higher above bars
            f"{height:.1f}%",
            ha="center",
            fontsize=9
        )

plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Clustered bar chart saved to {save_path}")

plt.show()

