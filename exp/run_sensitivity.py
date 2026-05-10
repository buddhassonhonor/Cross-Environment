import argparse
import json
import os
import time
import pandas as pd
import torch
from run_experiments_v2 import run_experiment

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rotated_mnist")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--data-root", default=os.path.join("exp", "data"))
    parser.add_argument("--results-path", default=os.path.join("exp", "results", "sensitivity.csv"))
    parser.add_argument("--max-mnist-train", type=int, default=20000)
    parser.add_argument("--max-mnist-test", type=int, default=5000)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Grid search
    lambda_inv_list = [0.1, 0.5, 1.0, 2.0, 5.0]
    lambda_ent_list = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    records = []
    start = time.time()
    
    for l_inv in lambda_inv_list:
        for l_ent in lambda_ent_list:
            for seed in args.seeds:
                print(f"Running IPDR with inv={l_inv}, ent={l_ent}, seed={seed}")
                hparams = {
                    "inv_lambda": l_inv,
                    "ent_lambda": l_ent
                }
                result = run_experiment(args.dataset, "ipdr", seed, device, args, hparams_override=hparams)
                result["lambda_inv"] = l_inv
                result["lambda_ent"] = l_ent
                records.append(result)

    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(args.results_path, index=False)
    print(f"Saved sensitivity results to {args.results_path}")

if __name__ == "__main__":
    main()
