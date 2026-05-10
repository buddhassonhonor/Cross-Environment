import argparse
import json
import os

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", default=os.path.join("exp", "results", "results.csv"))
    parser.add_argument("--out-path", default=os.path.join("exp", "results", "summary.csv"))
    parser.add_argument("--n-perm", type=int, default=20000)
    return parser.parse_args()


def paired_sign_flip_pvalue(diffs: np.ndarray, n_perm: int, seed: int = 0) -> float:
    diffs = np.asarray(diffs, dtype=np.float64)
    if diffs.size == 0:
        return float("nan")
    rng = np.random.RandomState(seed)
    observed = float(diffs.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diffs.size))
    perm_means = (signs * diffs).mean(axis=1)
    p = float((np.abs(perm_means) >= abs(observed)).mean())
    return p


def parse_curve(series: pd.Series) -> list[list[float]]:
    curves = []
    for val in series.fillna("[]").tolist():
        try:
            curves.append(json.loads(val))
        except json.JSONDecodeError:
            curves.append([])
    return curves


def write_diagnostics(df: pd.DataFrame, out_dir: str, dataset: str, methods: list[str]) -> None:
    subset = df[(df["dataset"] == dataset) & (df["method"].isin(methods))].copy()
    if subset.empty:
        return

    records = []
    for method in methods:
        part = subset[subset["method"] == method]
        ent_curves = parse_curve(part["weight_entropy_curve"])
        max_curves = parse_curve(part["weight_max_curve"])
        max_len = max((len(c) for c in ent_curves), default=0)
        for epoch in range(max_len):
            ent_vals = [c[epoch] for c in ent_curves if len(c) > epoch]
            max_vals = [c[epoch] for c in max_curves if len(c) > epoch]
            if not ent_vals:
                continue
            records.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "epoch": epoch + 1,
                    "entropy_mean": float(np.mean(ent_vals)),
                    "entropy_std": float(np.std(ent_vals, ddof=1)) if len(ent_vals) > 1 else 0.0,
                    "weight_max_mean": float(np.mean(max_vals)) if max_vals else float("nan"),
                    "weight_max_std": float(np.std(max_vals, ddof=1)) if len(max_vals) > 1 else 0.0,
                }
            )

    if not records:
        return

    diag = pd.DataFrame(records)
    diag_path = os.path.join(out_dir, f"diagnostics_{dataset}.csv")
    diag.to_csv(diag_path, index=False)

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "lines.linewidth": 2.0,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=220)
    for method in methods:
        part = diag[diag["method"] == method].sort_values("epoch")
        if part.empty:
            continue
        axes[0].plot(part["epoch"], part["entropy_mean"], label=method)
        axes[1].plot(part["epoch"], part["weight_max_mean"], label=method)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Entropy H(w)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("max_e w_e")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, f"diagnostics_{dataset}.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    df = pd.read_csv(args.results_path)

    out_dir = os.path.dirname(args.out_path)
    os.makedirs(out_dir, exist_ok=True)

    if "weight_entropy_curve" in df.columns:
        ent_curves = parse_curve(df["weight_entropy_curve"])
        max_curves = parse_curve(df["weight_max_curve"])
        df["weight_entropy_last"] = [c[-1] if c else float("nan") for c in ent_curves]
        df["weight_max_last"] = [c[-1] if c else float("nan") for c in max_curves]
        df["weight_entropy_mean"] = [float(np.mean(c)) if c else float("nan") for c in ent_curves]
        df["weight_max_mean"] = [float(np.mean(c)) if c else float("nan") for c in max_curves]

    grouped = (
        df.groupby(["dataset", "method"])
        .agg(mean_acc=("mean_acc", "mean"), std_acc=("mean_acc", "std"),
             mean_worst=("worst_acc", "mean"), std_worst=("worst_acc", "std"))
        .reset_index()
    )

    stability = (
        df.groupby(["dataset", "method"])
        .agg(
            weight_entropy_last=("weight_entropy_last", "mean"),
            weight_max_last=("weight_max_last", "mean"),
            weight_entropy_mean=("weight_entropy_mean", "mean"),
            weight_max_mean=("weight_max_mean", "mean"),
        )
        .reset_index()
    )
    grouped = grouped.merge(stability, on=["dataset", "method"], how="left")

    pvals = []
    for dataset in sorted(df["dataset"].unique().tolist()):
        wide = df[df["dataset"] == dataset].pivot_table(
            index="seed", columns="method", values="worst_acc", aggfunc="mean"
        )
        if "ipdr" in wide.columns and "groupdro" in wide.columns:
            paired = wide[["ipdr", "groupdro"]].dropna()
            diffs = (paired["ipdr"] - paired["groupdro"]).to_numpy()
            p = paired_sign_flip_pvalue(diffs, n_perm=args.n_perm, seed=0)
            pvals.append({"dataset": dataset, "p_ipdr_vs_groupdro_worst": p, "diff_ipdr_vs_groupdro_worst": float(np.mean(diffs))})
        else:
            pvals.append({"dataset": dataset, "p_ipdr_vs_groupdro_worst": float("nan"), "diff_ipdr_vs_groupdro_worst": float("nan")})

    pvals_df = pd.DataFrame(pvals)
    grouped.to_csv(args.out_path, index=False)


    pvals_path = os.path.join(out_dir, "significance.csv")
    pvals_df.to_csv(pvals_path, index=False)

    methods_for_diag = ["groupdro", "ipdr_no_ent", "ipdr_dro_inv", "ipdr_no_inv", "ipdr"]
    write_diagnostics(df, out_dir=out_dir, dataset="rotated_mnist", methods=methods_for_diag)
    write_diagnostics(df, out_dir=out_dir, dataset="synthetic_spurious", methods=methods_for_diag)

if __name__ == "__main__":
    main()
