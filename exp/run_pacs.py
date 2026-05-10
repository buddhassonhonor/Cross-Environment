import os
import subprocess
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # Configuration
    datasets = ["PACS"]
    algorithms = ["ERM", "GroupDRO", "IPDR"]
    test_envs_map = {
        "PACS": [0, 1, 2, 3],
    }
    seeds = [0, 1, 2] # 3 seeds for mean/std
    steps = 500 # Short run for feasibility on CPU
    data_dir = os.path.join("exp", "domainbed_data")
    output_dir_base = os.path.join("exp", "domainbed_results")
    
    # Ensure IPDR is in algorithms.py (I checked and it is there!)

    for dataset in datasets:
        test_envs = test_envs_map[dataset]
        for algorithm in algorithms:
            for test_env in test_envs:
                for seed in seeds:
                    output_dir = os.path.join(output_dir_base, dataset, algorithm, f"test_env{test_env}", f"seed{seed}")
                    
                    # Check if done
                    if os.path.exists(os.path.join(output_dir, "done")):
                        print(f"Skipping {output_dir} (already done)")
                        continue
                    
                    cmd = (
                        f"python -m domainbed.scripts.train "
                    f"--data_dir {data_dir} "
                    f"--algorithm {algorithm} "
                    f"--dataset {dataset} "
                    f"--test_envs {test_env} "
                    f"--seed {seed} "
                    f"--steps {steps} "
                    f"--checkpoint_freq {steps} " # Save at the end
                    f"--output_dir {output_dir} "
                )
                
                # Run from root directory so python module path works
                # Assuming script is run from d:\claude-code\math18\exp
                # But python -m domainbed requires d:\claude-code\math18\exp\domainbed in PYTHONPATH
                # or run from d:\claude-code\math18\exp\domainbed
                
                # Let's handle path
                # Current CWD is d:\claude-code\math18
                # domainbed is in exp/domainbed
                
                # The command needs to be adjusted.
                # If I run from d:\claude-code\math18\exp\domainbed
                
                cwd = os.path.join("exp", "domainbed")
                # Adjust data_dir relative to cwd
                # data_dir was exp/domainbed_data. Relative to exp/domainbed it is ../domainbed_data
                rel_data_dir = os.path.join("..", "domainbed_data")
                rel_output_dir = os.path.join("..", "domainbed_results", dataset, algorithm, f"test_env{test_env}", f"seed{seed}")
                
                cmd = (
                    f"python -m domainbed.scripts.train "
                    f"--data_dir {rel_data_dir} "
                    f"--algorithm {algorithm} "
                    f"--dataset {dataset} "
                    f"--test_envs {test_env} "
                    f"--seed {seed} "
                    f"--steps {steps} "
                    f"--checkpoint_freq {steps} "
                    f"--output_dir {rel_output_dir} "
                )
                
                print(f"Starting {algorithm} on {dataset} test_env {test_env}")
                try:
                    subprocess.run(cmd, shell=True, cwd=cwd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running {cmd}: {e}")

if __name__ == "__main__":
    main()
