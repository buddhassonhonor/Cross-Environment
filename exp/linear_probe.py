import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


class FeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        h = self.feature(x)
        logits = self.classifier(h)
        if return_features:
            return logits, h
        return logits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_label_conditional_penalty(
    features_list, labels_list, num_classes: int
) -> torch.Tensor:
    penalty = 0.0
    for cls in range(num_classes):
        env_means = []
        for feats, labels in zip(features_list, labels_list):
            mask = labels == cls
            if mask.sum() < 2:
                continue
            env_means.append(feats[mask].mean(dim=0))
        if len(env_means) <= 1:
            continue
        means = torch.stack(env_means, dim=0)
        penalty = penalty + ((means - means.mean(dim=0)) ** 2).mean()
    return penalty


def update_env_weights(
    weights: torch.Tensor, losses: torch.Tensor, eta: float, ent_lambda: float = 0.0
) -> torch.Tensor:
    new_weights = weights * torch.exp(eta * losses.detach())
    new_weights = new_weights / new_weights.sum()
    if ent_lambda > 0:
        uniform = torch.ones_like(new_weights) / len(new_weights)
        new_weights = (1.0 - ent_lambda) * new_weights + ent_lambda * uniform
    return new_weights


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def parse_args():
    parser = argparse.ArgumentParser(description="Linear probing on frozen features")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", default="PACS")
    parser.add_argument("--output-dir", default=os.path.join("exp", "linear_probe"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["erm", "groupdro", "ipdr_inv", "ipdr_no_ent", "ipdr"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--inv-lambda", type=float, default=1.0)
    parser.add_argument("--ent-lambda", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--weights-path", default=None)
    return parser.parse_args()


def get_env_dirs(data_dir: str, dataset: str):
    if dataset == "PACS":
        root = os.path.join(data_dir, "PACS")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    envs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    return root, envs


def build_backbone(device: torch.device, weights_path: str | None) -> nn.Module:
    if weights_path and os.path.exists(weights_path):
        backbone = models.resnet50(weights=None)
        state = torch.load(weights_path, map_location="cpu")
        backbone.load_state_dict(state)
    else:
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone.eval()
    backbone.to(device)
    return backbone


def extract_features(
    root: str,
    envs,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    output_dir: str,
    overwrite: bool,
    weights_path: str | None,
):
    os.makedirs(output_dir, exist_ok=True)
    backbone = build_backbone(device, weights_path)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    for env in envs:
        out_path = os.path.join(output_dir, f"{env}.pt")
        if os.path.exists(out_path) and not overwrite:
            continue
        dataset = ImageFolder(os.path.join(root, env), transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        features = []
        labels = []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                feats = backbone(images).cpu()
                features.append(feats)
                labels.append(targets)
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        torch.save({"features": features, "labels": labels}, out_path)


def train_and_eval(
    features_by_env,
    labels_by_env,
    test_env_idx: int,
    method: str,
    seed: int,
    args,
    device: torch.device,
):
    set_seed(seed)
    num_classes = int(labels_by_env[0].max().item() + 1)
    model = MLP(features_by_env[0].shape[1], args.hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_envs = []
    for env_i, (feats, labels) in enumerate(zip(features_by_env, labels_by_env)):
        if env_i == test_env_idx:
            continue
        train_envs.append(FeatureDataset(feats, labels))

    env_weights = torch.ones(len(train_envs), device=device) / len(train_envs)
    loaders = [
        DataLoader(
            env,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        for env in train_envs
    ]

    steps_per_epoch = min(len(loader) for loader in loaders)
    for _ in range(args.epochs):
        model.train()
        batches = zip(*loaders)
        for _ in range(steps_per_epoch):
            batch = next(batches)
            env_losses = []
            features_list = []
            labels_list = []
            for env_x, env_y in batch:
                env_x = env_x.to(device)
                env_y = env_y.to(device)
                logits, feats = model(env_x, return_features=True)
                loss = criterion(logits, env_y)
                env_losses.append(loss)
                features_list.append(feats)
                labels_list.append(env_y)

            losses = torch.stack(env_losses)
            inv_penalty = compute_label_conditional_penalty(
                features_list, labels_list, num_classes
            )

            if method == "erm":
                loss = losses.mean()
            elif method == "groupdro":
                env_weights = update_env_weights(env_weights, losses, args.eta)
                loss = (env_weights * losses).sum()
            elif method == "ipdr_inv":
                loss = losses.mean() + args.inv_lambda * inv_penalty
            elif method == "ipdr_no_ent":
                env_weights = update_env_weights(env_weights, losses, args.eta)
                loss = (env_weights * losses).sum() + args.inv_lambda * inv_penalty
            elif method == "ipdr":
                env_weights = update_env_weights(
                    env_weights, losses, args.eta, args.ent_lambda
                )
                loss = (env_weights * losses).sum() + args.inv_lambda * inv_penalty
            else:
                raise ValueError(f"Unknown method: {method}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    test_feats = features_by_env[test_env_idx].to(device)
    test_labels = labels_by_env[test_env_idx].to(device)
    with torch.no_grad():
        logits = model(test_feats)
    acc = accuracy_from_logits(logits, test_labels)
    return acc


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_weights = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "torch",
        "hub",
        "checkpoints",
        "resnet50-0676ba61.pth",
    )
    if args.weights_path is None and os.path.exists(default_weights):
        args.weights_path = default_weights
    root, envs = get_env_dirs(args.data_dir, args.dataset)

    feat_dir = os.path.join(args.output_dir, "features", args.dataset)
    extract_features(
        root,
        envs,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=feat_dir,
        overwrite=args.overwrite_features,
        weights_path=args.weights_path,
    )

    features_by_env = []
    labels_by_env = []
    for env in envs:
        payload = torch.load(os.path.join(feat_dir, f"{env}.pt"))
        features_by_env.append(payload["features"])
        labels_by_env.append(payload["labels"])

    results_path = os.path.join(args.output_dir, f"{args.dataset.lower()}_results.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    write_header = (not os.path.exists(results_path)) or os.path.getsize(results_path) == 0
    start = time.time()
    with open(results_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "method", "seed", "test_env", "acc"]
        )
        if write_header:
            writer.writeheader()
        for test_env_idx, env_name in enumerate(envs):
            for method in args.methods:
                for seed in args.seeds:
                    acc = train_and_eval(
                        features_by_env,
                        labels_by_env,
                        test_env_idx,
                        method,
                        seed,
                        args,
                        device,
                    )
                    writer.writerow(
                        {
                            "dataset": args.dataset,
                            "method": method,
                            "seed": seed,
                            "test_env": env_name,
                            "acc": acc,
                        }
                    )
                    handle.flush()
                    print(
                        f"[{args.dataset}][{method}][seed={seed}] "
                        f"test_env={env_name} acc={acc:.4f}"
                    )

    meta_path = os.path.join(args.output_dir, f"{args.dataset.lower()}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "envs": envs,
                "methods": args.methods,
                "seeds": args.seeds,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "device": str(device),
                "elapsed_sec": time.time() - start,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
