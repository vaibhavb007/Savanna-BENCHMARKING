import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

# ---------------- Style ---------------- #
plt.style.use("ggplot")

# ---------------- Helpers ---------------- #

def load_config(path):
    with open(path) as f:
        return json.load(f)

def load_csv(results_dir, filename):
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close(fig)

def get_run_dirs(base_results):
    pattern = os.path.join(base_results, "Run*")
    dirs = glob.glob(pattern)
    dirs = [d for d in dirs if os.path.isdir(d) and os.path.basename(d).startswith("Run")]
    return sorted(dirs)

def load_all_runs(base_results, filename):
    run_dirs = get_run_dirs(base_results)
    dfs = []
    for rd in run_dirs:
        df = load_csv(rd, filename)
        if df is not None:
            dfs.append(df)
    return dfs

def categorize_question(text):
    t = str(text).lower()
    if "equal to" in t: return "Equality"
    elif "more" in t or "less" in t: return "Comparison"
    elif any(w in t for w in ["many", "number", "amount"]): return "Counting"
    elif any(w in t for w in ["small", "large", "square", "circle"]): return "Attribute"
    elif " at the " in t: return "Location"
    elif t.startswith(("is", "are", "does", "do")): return "Yes/No"
    return "Other"

def extract_family_and_model(filename):
    name = os.path.splitext(filename)[0]
    lower = name.lower()
    if "gemma3" in lower: family = "Gemma3"
    elif "ministral" in lower: family = "Ministral3"
    elif "qwen2" in lower: family = "Qwen2"
    elif "llava" in lower: family = "Llava"
    else: family = "Other"
    return family, name

def get_color_map(cfg):
    """Assign colors to each model. Prevent dupes"""
    all_models = []
    for engine, models in cfg["models"].items():
        for model_name in models.keys():
            all_models.append(model_name)
    
    unique_models = sorted(list(set(all_models))) 
    custom_colors = [
        '#1A73E8', '#D93025', '#188038', '#F29900', '#A142F4', '#00ACC1', 
        '#FF4081', '#5F6368', '#E67C73', '#111111', '#7FDBFF', '#39CCCC', 
        '#3D9970', '#2ECC40', '#01FF70', '#FFDC00', '#FF851B', '#FF4136', 
        '#85144B', '#F012BE', '#B10DC9', '#0074D9', '#311B92', '#004D40',
        '#827717', '#3E2723'
    ]
    color_dict = {}
    for i, model in enumerate(unique_models):
        color_dict[model] = custom_colors[i % len(custom_colors)]
        
    return color_dict

# ---------------- Single‑run plots ---------------- #

def plot_accuracy(cfg, results_dir, plots_dir):
    for engine, models in cfg["models"].items():
        rows = []
        for model, file in models.items():
            df = load_csv(results_dir, file)
            if df is not None:
                acc = df["correct"].mean() * 100
                rows.append((model, acc))
        if not rows: continue
        df_table = pd.DataFrame(rows, columns=["Model", "Accuracy"]).sort_values("Accuracy", ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        table = ax.table(cellText=[[m, f"{a:.2f}%"] for m, a in df_table.values],
                         colLabels=["Model", "Accuracy"], cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        ax.set_title(f"{engine} - Overall Accuracy", fontsize=14, weight="bold")
        save(fig, plots_dir, f"accuracy_table_{engine}.png")

def plot_accuracy_by_type(cfg, results_dir, plots_dir, color_map):
    for engine, models in cfg["models"].items():
        all_data = []
        for model, filename in models.items():
            df = load_csv(results_dir, filename)
            if df is not None:
                df["qtype"] = df["question_text"].apply(categorize_question)
                grouped = df.groupby("qtype")["correct"].mean() * 100
                grouped.name = model
                all_data.append(grouped)
        if not all_data: continue
        combined_df = pd.concat(all_data, axis=1).fillna(0)
        x = np.arange(len(combined_df.index))
        width = 0.8 / len(combined_df.columns)
        fig, ax = plt.subplots(figsize=(12, 7))
        for i, model in enumerate(combined_df.columns):
            offset = (i - (len(combined_df.columns) - 1) / 2) * width
            ax.bar(x + offset, combined_df[model], width, label=model, color=color_map.get(model))
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{engine} - Accuracy by Question Type', weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(combined_df.index, rotation=15)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
        save(fig, plots_dir, f"accuracy_by_type_{engine}.png")

def plot_latency(cfg, results_dir, plots_dir, color_map):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, file in models.items():
            df = load_csv(results_dir, file)
            if df is not None:
                # Skip first
                smoothed = df["latency_sec"].iloc[1:].rolling(window=10).mean()
                ax.plot(smoothed, label=model, linewidth=2, color=color_map.get(model))
        ax.set_ylabel("Latency (s)")
        ax.set_title(f"{engine} - Latency per Prompt", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"latency_{engine}.png")

def plot_power(cfg, results_dir, plots_dir, color_map):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, file in models.items():
            df = load_csv(results_dir, file)
            if df is not None and "avg_gpu_w" in df.columns:
                smoothed = df["avg_gpu_w"].rolling(10, min_periods=1).mean()
                ax.plot(smoothed, label=model, linewidth=2, color=color_map.get(model))
        ax.set_ylabel("GPU Power (W)")
        ax.set_title(f"{engine} - Average GPU Power", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"power_gpu_{engine}.png")

def plot_temperature(cfg, results_dir, plots_dir, color_map):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, file in models.items():
            df = load_csv(results_dir, file)
            if df is not None and "max_gpu_temp_c" in df.columns:
                smoothed = df["max_gpu_temp_c"].rolling(10, min_periods=1).mean()
                ax.plot(smoothed, label=model, linewidth=2, color=color_map.get(model))
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"{engine} - Max Temperature", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"temperature_{engine}.png")

def plot_accuracy_vs_temp(cfg, results_dir, plots_dir, color_map):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, file in models.items():
            df = load_csv(results_dir, file)
            if df is not None and "max_gpu_temp_c" in df.columns:
                df["temp_bin"] = df["max_gpu_temp_c"].round()
                temp_acc = df.groupby("temp_bin")["correct"].mean() * 100
                ax.plot(temp_acc.index, temp_acc.values, marker='o', label=model, color=color_map.get(model))
        ax.set_xlabel("GPU Temperature (°C)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{engine} - Accuracy Trend as Temperature Rises", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"acc_vs_temp_{engine}.png")

def plot_energy(cfg, results_dir, plots_dir, color_map):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, file in models.items():
            df = load_csv(results_dir, file)
            if df is not None and "avg_gpu_w" in df.columns and "latency_sec" in df.columns:
                # Skip first
                energy = (df["avg_gpu_w"] * df["latency_sec"]).iloc[1:]
                smoothed = energy.rolling(window=10, min_periods=1).mean()
                ax.plot(smoothed, label=model, linewidth=2, color=color_map.get(model))
        ax.set_ylabel("Energy (Joules)")
        ax.set_xlabel("Prompt Index")
        ax.set_title(f"{engine} - Energy per Prompt (Moving Avg)", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"energy_per_prompt_{engine}.png")

# ---------------- Aggregated plots (Engine‑based) ---------------- #

def plot_accuracy_aggregated(cfg, results_dir, plots_dir):
    for engine, models in cfg["models"].items():
        rows = []
        for model, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs: continue
            run_accs = [df["correct"].mean() * 100 for df in dfs]
            rows.append((model, np.mean(run_accs), np.std(run_accs)))
        if not rows: continue
        rows.sort(key=lambda x: x[1], reverse=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        cell_text = [[m, rf"{mu:.2f}% $\pm$ {sd:.2f}"] for m, mu, sd in rows]
        table = ax.table(cell_text, colLabels=["Model", r"Accuracy (Mean $\pm$ Std)"], cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        ax.set_title(f"{engine} - Overall Aggregated Accuracy", fontsize=14, weight="bold")
        save(fig, plots_dir, f"accuracy_overall_{engine}.png")

def plot_accuracy_by_type_aggregated(cfg, results_dir, plots_dir, color_map):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(12, 7))
        model_names = list(models.keys())
        all_types = sorted(["Equality", "Comparison", "Counting", "Attribute", "Location", "Yes/No"])
        x = np.arange(len(all_types))
        width = 0.8 / len(model_names)
        for i, model in enumerate(model_names):
            dfs = load_all_runs(results_dir, models[model])
            type_accs = {t: [] for t in all_types}
            for df in dfs:
                df["qtype"] = df["question_text"].apply(categorize_question)
                grouped = df.groupby("qtype")["correct"].mean() * 100
                for t in all_types:
                    if t in grouped: type_accs[t].append(grouped[t])
            means = [np.mean(type_accs[t]) if type_accs[t] else 0 for t in all_types]
            stds = [np.std(type_accs[t]) if type_accs[t] else 0 for t in all_types]
            offset = (i - (len(model_names) - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=model, color=color_map.get(model))
        ax.set_xticks(x)
        ax.set_xticklabels(all_types, rotation=15)
        ax.set_title(f"{engine} - Accuracy by Question Type (Mean ± Std)", weight="bold")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
        save(fig, plots_dir, f"accuracy_by_type_{engine}.png")

def plot_latency_aggregated(cfg, results_dir, plots_dir, color_map):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs: continue
            processed = [df["latency_sec"].iloc[1:].values for df in dfs]
            min_len = min(len(p) for p in processed)
            arr = np.array([p[:min_len] for p in processed])
            mean_lat = pd.Series(np.mean(arr, axis=0)).rolling(10, min_periods=1).mean()
            std_lat = pd.Series(np.std(arr, axis=0)).rolling(10, min_periods=1).mean()
            iters = np.arange(1, min_len + 1)
            color = color_map.get(model)
            line, = ax.plot(iters, mean_lat, label=model, linewidth=2, color=color)
            ax.fill_between(iters, mean_lat - std_lat, mean_lat + std_lat, alpha=0.2, color=color)
        ax.set_title(f"{engine} - Latency (Mean ± Std)", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"latency_{engine}.png")

def plot_power_aggregated(cfg, results_dir, plots_dir, color_map):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs or "avg_gpu_w" not in dfs[0].columns: continue
            processed = [df["avg_gpu_w"].values for df in dfs]
            min_len = min(len(p) for p in processed)
            arr = np.array([p[:min_len] for p in processed])
            mean_p = pd.Series(np.mean(arr, axis=0)).rolling(10, min_periods=1).mean()
            std_p = pd.Series(np.std(arr, axis=0)).rolling(10, min_periods=1).mean()
            color = color_map.get(model)
            line, = ax.plot(mean_p, label=model, linewidth=2, color=color)
            ax.fill_between(range(min_len), mean_p - std_p, mean_p + std_p, alpha=0.2, color=color)
        ax.set_title(f"{engine} - GPU Power (Mean ± Std)", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"power_gpu_{engine}.png")

def plot_temperature_aggregated(cfg, results_dir, plots_dir, color_map):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs or "max_gpu_temp_c" not in dfs[0].columns: continue
            processed = [df["max_gpu_temp_c"].values for df in dfs]
            min_len = min(len(p) for p in processed)
            arr = np.array([p[:min_len] for p in processed])
            mean_t = pd.Series(np.mean(arr, axis=0)).rolling(10, min_periods=1).mean()
            std_t = pd.Series(np.std(arr, axis=0)).rolling(10, min_periods=1).mean()
            color = color_map.get(model)
            line, = ax.plot(mean_t, label=model, linewidth=2, color=color)
            ax.fill_between(range(min_len), mean_t - std_t, mean_t + std_t, alpha=0.2, color=color)
        ax.set_title(f"{engine} - Max Temperature (Mean ± Std)", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"temperature_{engine}.png")

def plot_accuracy_vs_temp_aggregated(cfg, results_dir, plots_dir, color_map):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs or "max_gpu_temp_c" not in dfs[0].columns: continue
            all_df = pd.concat(dfs, ignore_index=True)
            all_df["temp_bin"] = all_df["max_gpu_temp_c"].round()
            grouped = all_df.groupby("temp_bin")["correct"].agg(['mean', 'std']) * 100
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], marker='o', capsize=3, label=model, color=color_map.get(model))
        ax.set_title(f"{engine} - Accuracy vs Temperature (Mean ± Std)", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"acc_vs_temp_{engine}.png")

def plot_energy_cdf_aggregated(cfg, results_dir, plots_dir, color_map):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        has_any = False
        for model_name, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs or "avg_gpu_w" not in dfs[0].columns or "latency_sec" not in dfs[0].columns: continue
            all_energy = []
            for df in dfs:
                energy_joules = df["avg_gpu_w"] * df["latency_sec"]
                all_energy.extend(energy_joules.dropna().values)
            if not all_energy: continue
            has_any = True
            sorted_e = np.sort(all_energy)
            cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
            ax.step(sorted_e, cdf, where='post', label=model_name, linewidth=2, color=color_map.get(model_name))
        if not has_any: continue
        ax.set_title(f"{engine} - Energy per Query CDF", weight="bold")
        ax.set_xscale('log')
        ax.set_xlabel("Energy (Joules)")
        ax.set_ylabel("Cumulative Probability")
        ax.legend(loc='lower right')
        save(fig, plots_dir, f"energy_cdf_{engine}.png")

def plot_energy_aggregated(cfg, results_dir, plots_dir, color_map):
    for engine, models in cfg["models"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, filename in models.items():
            dfs = load_all_runs(results_dir, filename)
            if not dfs or "avg_gpu_w" not in dfs[0].columns or "latency_sec" not in dfs[0].columns: 
                continue
            processed = [(df["avg_gpu_w"] * df["latency_sec"]).iloc[1:].values for df in dfs]
            min_len = min(len(p) for p in processed)
            arr = np.array([p[:min_len] for p in processed])
            mean_e = pd.Series(np.mean(arr, axis=0)).rolling(10, min_periods=1).mean()
            std_e = pd.Series(np.std(arr, axis=0)).rolling(10, min_periods=1).mean()
            iters = np.arange(1, min_len + 1)
            color = color_map.get(model)
            ax.plot(iters, mean_e, label=model, linewidth=2, color=color)
            ax.fill_between(iters, mean_e - std_e, mean_e + std_e, alpha=0.2, color=color)
        ax.set_ylabel("Energy (Joules)")
        ax.set_title(f"{engine} - Energy per Prompt (Mean ± Std)", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"energy_per_prompt_{engine}.png")

# ---------------- Family‑based aggregated plots ---------------- #

def load_all_data_by_family(base_results, config):
    family_data = {}
    run_dirs = get_run_dirs(base_results)
    for run_dir in run_dirs:
        for engine, models in config["models"].items():
            for model_name, filename in models.items():
                df = load_csv(run_dir, filename)
                if df is None: continue
                family, _ = extract_family_and_model(filename)
                if family not in family_data: family_data[family] = {}
                if model_name not in family_data[family]: family_data[family][model_name] = []
                family_data[family][model_name].append(df)
    return family_data

def plot_accuracy_family(family_data, plots_dir):
    for family, models in family_data.items():
        rows = []
        for model_name, dfs in models.items():
            all_acc = []
            for df in dfs: all_acc.extend(df["correct"].values)
            mean_acc = np.mean(all_acc) * 100
            std_acc = np.std(all_acc) * 100
            rows.append((model_name, mean_acc, std_acc))
        if not rows: continue
        rows.sort(key=lambda x: x[1], reverse=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        cell_text = [[m, f"{mu:.2f}% ± {sd:.2f}"] for m, mu, sd in rows]
        table = ax.table(cell_text, colLabels=["Model", "Accuracy (Mean ± Std)"], cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        ax.set_title(f"{family} - Overall Aggregated Accuracy", fontsize=14, weight="bold")
        save(fig, plots_dir, f"accuracy_overall_{family}.png")

def plot_accuracy_by_type_family(family_data, plots_dir, color_map):
    all_types = ["Equality", "Comparison", "Counting", "Attribute", "Location", "Yes/No"]
    for family, models in family_data.items():
        fig, ax = plt.subplots(figsize=(12, 7))
        model_names = list(models.keys())
        x = np.arange(len(all_types))
        width = 0.8 / len(model_names) if model_names else 0.8
        for i, model_name in enumerate(model_names):
            dfs = models[model_name]
            type_accs = {t: [] for t in all_types}
            for df in dfs:
                df = df.copy()
                df["qtype"] = df["question_text"].apply(categorize_question)
                grouped = df.groupby("qtype")["correct"].mean() * 100
                for t in all_types:
                    if t in grouped: type_accs[t].append(grouped[t])
            means = [np.mean(type_accs[t]) if type_accs[t] else 0 for t in all_types]
            stds = [np.std(type_accs[t]) if type_accs[t] else 0 for t in all_types]
            offset = (i - (len(model_names) - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=model_name, color=color_map.get(model_name))
        ax.set_xticks(x)
        ax.set_xticklabels(all_types, rotation=15)
        ax.set_title(f"{family} - Accuracy by Question Type (Mean ± Std)", weight="bold")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
        save(fig, plots_dir, f"accuracy_by_type_{family}.png")

def plot_latency_family(family_data, plots_dir, color_map):
    for family, models in family_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_name, dfs in models.items():
            lat_series = [df["latency_sec"].iloc[1:].values for df in dfs]
            if not lat_series: continue
            min_len = min(len(ls) for ls in lat_series)
            arr = np.array([ls[:min_len] for ls in lat_series])
            mean_lat = pd.Series(np.mean(arr, axis=0)).rolling(10, min_periods=1).mean()
            std_lat = pd.Series(np.std(arr, axis=0)).rolling(10, min_periods=1).mean()
            iters = np.arange(1, min_len + 1)
            color = color_map.get(model_name)
            line, = ax.plot(iters, mean_lat, label=model_name, linewidth=2, color=color)
            ax.fill_between(iters, mean_lat - std_lat, mean_lat + std_lat, alpha=0.2, color=color)
        ax.set_title(f"{family} - Latency (Mean ± Std)", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"latency_{family}.png")

def plot_power_family(family_data, plots_dir, color_map):
    for family, models in family_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_name, dfs in models.items():
            if "avg_gpu_w" not in dfs[0].columns: continue
            power_series = [df["avg_gpu_w"].values for df in dfs]
            min_len = min(len(ps) for ps in power_series)
            arr = np.array([ps[:min_len] for ps in power_series])
            mean_p = pd.Series(np.mean(arr, axis=0)).rolling(10, min_periods=1).mean()
            std_p = pd.Series(np.std(arr, axis=0)).rolling(10, min_periods=1).mean()
            color = color_map.get(model_name)
            line, = ax.plot(mean_p, label=model_name, linewidth=2, color=color)
            ax.fill_between(range(min_len), mean_p - std_p, mean_p + std_p, alpha=0.2, color=color)
        ax.set_title(f"{family} - GPU Power (Mean ± Std)", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"power_gpu_{family}.png")

def plot_temperature_family(family_data, plots_dir, color_map):
    for family, models in family_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_name, dfs in models.items():
            if "max_gpu_temp_c" not in dfs[0].columns: continue
            temp_series = [df["max_gpu_temp_c"].values for df in dfs]
            min_len = min(len(ts) for ts in temp_series)
            arr = np.array([ts[:min_len] for ts in temp_series])
            mean_t = pd.Series(np.mean(arr, axis=0)).rolling(10, min_periods=1).mean()
            std_t = pd.Series(np.std(arr, axis=0)).rolling(10, min_periods=1).mean()
            color = color_map.get(model_name)
            line, = ax.plot(mean_t, label=model_name, linewidth=2, color=color)
            ax.fill_between(range(min_len), mean_t - std_t, mean_t + std_t, alpha=0.2, color=color)
        ax.set_title(f"{family} - Max Temperature (Mean ± Std)", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"temperature_{family}.png")

def plot_accuracy_vs_temp_family(family_data, plots_dir, color_map):
    for family, models in family_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_name, dfs in models.items():
            if "max_gpu_temp_c" not in dfs[0].columns: continue
            all_df = pd.concat(dfs, ignore_index=True)
            all_df["temp_bin"] = all_df["max_gpu_temp_c"].round()
            grouped = all_df.groupby("temp_bin")["correct"].agg(['mean', 'std']) * 100
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], marker='o', capsize=3, label=model_name, color=color_map.get(model_name))
        ax.set_title(f"{family} - Accuracy vs Temperature (Mean ± Std)", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"acc_vs_temp_{family}.png")

def plot_energy_cdf_family(family_data, plots_dir, color_map):
    for family, models in family_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        has_any = False
        for model_name, dfs in models.items():
            if not dfs or "avg_gpu_w" not in dfs[0].columns or "latency_sec" not in dfs[0].columns: continue
            all_energy = []
            for df in dfs:
                energy_joules = df["avg_gpu_w"] * df["latency_sec"]
                all_energy.extend(energy_joules.dropna().values)
            if not all_energy: continue
            has_any = True
            sorted_e = np.sort(all_energy)
            cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
            ax.step(sorted_e, cdf, where='post', label=model_name, linewidth=2, color=color_map.get(model_name))
        if not has_any: continue
        ax.set_title(f"{family} - Energy per Query CDF", weight="bold")
        ax.set_xscale('log')
        ax.legend(loc='lower right')
        save(fig, plots_dir, f"energy_cdf_{family}.png")

def plot_energy_family(family_data, plots_dir, color_map):
    for family, models in family_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_name, dfs in models.items():
            if "avg_gpu_w" not in dfs[0].columns or "latency_sec" not in dfs[0].columns: 
                continue
            energy_series = [(df["avg_gpu_w"] * df["latency_sec"]).iloc[1:].values for df in dfs]
            min_len = min(len(es) for es in energy_series)
            arr = np.array([es[:min_len] for es in energy_series])
            mean_e = pd.Series(np.mean(arr, axis=0)).rolling(10, min_periods=1).mean()
            std_e = pd.Series(np.std(arr, axis=0)).rolling(10, min_periods=1).mean()
            iters = np.arange(1, min_len + 1)
            color = color_map.get(model_name)
            ax.plot(iters, mean_e, label=model_name, linewidth=2, color=color)
            ax.fill_between(iters, mean_e - std_e, mean_e + std_e, alpha=0.2, color=color)
        ax.set_ylabel("Energy (Joules)")
        ax.set_title(f"{family} - Energy per Prompt (Mean ± Std)", weight="bold")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        save(fig, plots_dir, f"energy_per_prompt_{family}.png")
        
# ---------------- Main ---------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run", help="Run subdirectory (e.g., Run1)")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--family", action="store_true")
    p.add_argument("--accuracy", action="store_true")
    p.add_argument("--accuracy-by-type", action="store_true")
    p.add_argument("--latency", action="store_true")
    p.add_argument("--power", action="store_true")
    p.add_argument("--temp", action="store_true")
    p.add_argument("--acc-vs-temp", action="store_true")
    p.add_argument("--energy", action="store_true", help="Plot Energy per prompt time-series")
    p.add_argument("--energy-cdf", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    cmap = get_color_map(cfg)
    base_results = cfg["results_dir"]
    base_plots = cfg["plots_dir"]

    # Single Run Logic
    if not args.aggregate and not args.family:
        results_dir = os.path.join(base_results, args.run) if args.run else base_results
        plots_dir = os.path.join(base_plots, args.run) if args.run else base_plots
        if args.accuracy or args.all: plot_accuracy(cfg, results_dir, plots_dir)
        if args.accuracy_by_type or args.all: plot_accuracy_by_type(cfg, results_dir, plots_dir, cmap)
        if args.latency or args.all: plot_latency(cfg, results_dir, plots_dir, cmap)
        if args.power or args.all: plot_power(cfg, results_dir, plots_dir, cmap)
        if args.temp or args.all: plot_temperature(cfg, results_dir, plots_dir, cmap)
        if args.acc_vs_temp or args.all: plot_accuracy_vs_temp(cfg, results_dir, plots_dir, cmap)
        if args.energy or args.all: plot_energy(cfg, results_dir, plots_dir, cmap)
        return

    # Aggregated Logic
    if args.aggregate:
        plots_dir = os.path.join(base_plots, "aggregated")
        if args.accuracy or args.all: plot_accuracy_aggregated(cfg, base_results, plots_dir)
        if args.accuracy_by_type or args.all: plot_accuracy_by_type_aggregated(cfg, base_results, plots_dir, cmap)
        if args.latency or args.all: plot_latency_aggregated(cfg, base_results, plots_dir, cmap)
        if args.power or args.all: plot_power_aggregated(cfg, base_results, plots_dir, cmap)
        if args.temp or args.all: plot_temperature_aggregated(cfg, base_results, plots_dir, cmap)
        if args.acc_vs_temp or args.all: plot_accuracy_vs_temp_aggregated(cfg, base_results, plots_dir, cmap)
        if args.energy or args.all: plot_energy_aggregated(cfg, base_results, plots_dir, cmap)
        if args.energy_cdf or args.all: plot_energy_cdf_aggregated(cfg, base_results, plots_dir, cmap)

    # Family Logic
    if args.family:
        plots_dir = os.path.join(base_plots, "family")
        family_data = load_all_data_by_family(base_results, cfg)
        if args.accuracy or args.all: plot_accuracy_family(family_data, plots_dir)
        if args.accuracy_by_type or args.all: plot_accuracy_by_type_family(family_data, plots_dir, cmap)
        if args.latency or args.all: plot_latency_family(family_data, plots_dir, cmap)
        if args.power or args.all: plot_power_family(family_data, plots_dir, cmap)
        if args.temp or args.all: plot_temperature_family(family_data, plots_dir, cmap)
        if args.acc_vs_temp or args.all: plot_accuracy_vs_temp_family(family_data, plots_dir, cmap)
        if args.energy or args.all: plot_energy_family(family_data, plots_dir, cmap)
        if args.energy_cdf or args.all: plot_energy_cdf_family(family_data, plots_dir, cmap)

if __name__ == "__main__":
    main()
