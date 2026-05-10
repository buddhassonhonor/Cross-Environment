import os
import json
import pandas as pd

def main():
    results_dir = os.path.join("exp", "domainbed_results", "PACS")
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found.")
        return

    algorithms = ["ERM", "GroupDRO", "IPDR"]
    envs = [0, 1, 2, 3]
    seeds = [0, 1, 2]
    
    # We want mean +- std across seeds for each algorithm and env (Art, Cartoon, Photo, Sketch)
    # envs: 0: Art, 1: Cartoon, 2: Photo, 3: Sketch
    
    env_names = {0: "Art", 1: "Cartoon", 2: "Photo", 3: "Sketch"}
    
    records = []
    
    for algo in algorithms:
        for env in envs:
            for seed in seeds:
                file_path = os.path.join(results_dir, algo, f"test_env{env}", f"seed{seed}", "results.jsonl")
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        lines = f.readlines()
                        if lines:
                            last_line = json.loads(lines[-1])
                            # the accuracy on the test environment
                            acc_key = f"env{env}_out_acc"
                            if acc_key in last_line:
                                acc = last_line[acc_key]
                                records.append({"Algorithm": algo, "Env": env_names[env], "Seed": seed, "Acc": acc})
    
    if not records:
        print("No results found yet.")
        return

    df = pd.DataFrame(records)
    
    # Calculate mean and std
    summary = df.groupby(["Algorithm", "Env"])["Acc"].agg(['mean', 'std']).reset_index()
    
    # Also calculate overall average across envs per algorithm
    overall = df.groupby(["Algorithm", "Seed"])["Acc"].mean().reset_index()
    overall_summary = overall.groupby("Algorithm")["Acc"].agg(['mean', 'std']).reset_index()
    overall_summary["Env"] = "Avg"
    
    summary = pd.concat([summary, overall_summary], ignore_index=True)
    
    # Pivot to match table format
    pivot_mean = summary.pivot(index="Algorithm", columns="Env", values="mean")
    pivot_std = summary.pivot(index="Algorithm", columns="Env", values="std")
    
    cols = ["Art", "Cartoon", "Photo", "Sketch", "Avg"]
    
    print("\\begin{table}[!htbp]")
    print("  \\centering")
    print("  \\footnotesize")
    print("  \\caption{PACS full fine-tuning accuracy (mean $\\pm$ std across seeds).}")
    print("  \\label{tab:pacs-full}")
    print("  \\begin{tabular}{lccccc}")
    print("    \\toprule")
    print("    Method & Art & Cartoon & Photo & Sketch & Avg \\\\")
    print("    \\midrule")
    
    for algo in algorithms:
        if algo not in pivot_mean.index:
            continue
        row_str = f"    {algo}"
        for c in cols:
            if c in pivot_mean.columns and not pd.isna(pivot_mean.loc[algo, c]):
                m = pivot_mean.loc[algo, c]
                s = pivot_std.loc[algo, c]
                row_str += f" & {m:.3f} $\\pm$ {s:.3f}"
            else:
                row_str += " & - "
        row_str += " \\\\"
        print(row_str)
        
    print("    \\bottomrule")
    print("  \\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    main()
