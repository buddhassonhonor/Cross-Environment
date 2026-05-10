import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    results_path = os.path.join("exp", "results", "sensitivity.csv")
    if not os.path.exists(results_path):
        print(f"No sensitivity results found at {results_path}")
        return

    df = pd.read_csv(results_path)
    
    # Pivot for heatmap
    pivot_table = df.pivot(index="lambda_inv", columns="lambda_ent", values="worst_acc")
    
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )
    plt.figure(figsize=(10.5, 8.2), dpi=220)
    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        annot_kws={"fontsize": 14},
        cbar_kws={"label": "Worst-Group Accuracy"},
    )
    ax.set_title("Sensitivity of Worst-Group Accuracy to Hyperparameters")
    ax.set_xlabel("Entropy Regularization (lambda_ent)")
    ax.set_ylabel("Invariance Penalty (lambda_inv)")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    
    output_path = os.path.join("figures", "sensitivity_heatmap.png")
    os.makedirs("figures", exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved sensitivity heatmap to {output_path}")

if __name__ == "__main__":
    main()
