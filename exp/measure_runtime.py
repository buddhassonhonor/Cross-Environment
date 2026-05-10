import time
import torch
from run_experiments_v2 import MLP, build_rotated_mnist_envs, build_loaders, train_epoch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "exp/data/mnist"
    train_envs, _, _, num_classes = build_rotated_mnist_envs(data_root, 0, 20000, 5000)
    batch_size = 256
    train_loaders = build_loaders(train_envs, batch_size=batch_size, shuffle=True)
    
    methods = ["erm", "groupdro", "coral", "swad", "ipdr"]
    
    print("Method,RuntimePerEpoch(s)")
    for method in methods:
        model = MLP(1*28*28, 256, 128, num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        hparams = {"gdro_eta":0.1, "coral_lambda":1.0, "inv_lambda":1.0, "ent_lambda":0.1}
        env_weights = torch.ones(len(train_envs), device=device) / len(train_envs)
        
        # Warmup
        train_epoch(model, train_loaders, optimizer, criterion, method, hparams, num_classes, device, env_weights)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(5):
            _, env_weights, _ = train_epoch(model, train_loaders, optimizer, criterion, method, hparams, num_classes, device, env_weights)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        print(f"{method},{(end-start)/5.0:.4f}")

if __name__ == "__main__":
    main()