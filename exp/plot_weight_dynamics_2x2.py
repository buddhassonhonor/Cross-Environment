import os

import matplotlib.pyplot as plt
import pandas as pd


METHODS = [
    ("groupdro", "GroupDRO"),
    ("ipdr_no_ent", "IPDR-NoEnt"),
    ("ipdr_dro_inv", "IPDR-DRO+Inv"),
    ("ipdr_no_inv", "IPDR-NoInv"),
    ("ipdr", "IPDR"),
]


def plot_metric(ax, df: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    for method_key, method_name in METHODS:
        part = df[df["method"] == method_key].sort_values("epoch")
        if part.empty:
            continue
        ax.plot(part["epoch"], part[metric], label=method_name)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.6)


def main() -> None:
    rotated_path = os.path.join("exp", "results", "diagnostics_rotated_mnist.csv")
    synthetic_path = os.path.join("exp", "results", "diagnostics_synthetic_spurious.csv")
    output_path = os.path.join("figures", "diagnostics_weight_dynamics_2x2.png")

    if not os.path.exists(rotated_path) or not os.path.exists(synthetic_path):
        print("Diagnostics CSV not found. Please generate diagnostics first.")
        return

    rot = pd.read_csv(rotated_path)
    syn = pd.read_csv(synthetic_path)

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "lines.linewidth": 2.5,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.5), dpi=220, sharex=False)

    plot_metric(
        axes[0, 0],
        rot,
        "entropy_mean",
        "Rotated MNIST: Entropy $H(w)$",
        "Entropy",
    )
    plot_metric(
        axes[0, 1],
        rot,
        "weight_max_mean",
        "Rotated MNIST: Max Weight $\\max_e w_e$",
        "Max Weight",
    )
    plot_metric(
        axes[1, 0],
        syn,
        "entropy_mean",
        "Synthetic Spurious: Entropy $H(w)$",
        "Entropy",
    )
    plot_metric(
        axes[1, 1],
        syn,
        "weight_max_mean",
        "Synthetic Spurious: Max Weight $\\max_e w_e$",
        "Max Weight",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 2x2 diagnostics figure to {output_path}")


if __name__ == "__main__":
    main()
