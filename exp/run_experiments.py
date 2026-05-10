import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import functional as TF


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TensorDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


class RotatedMNISTSubset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, indices, angle: float) -> None:
        self.images = images[indices]
        self.labels = labels[indices]
        self.angle = angle

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        img = self.images[idx].float() / 255.0
        img = img.unsqueeze(0)
        img = TF.rotate(img, self.angle, interpolation=TF.InterpolationMode.BILINEAR)
        return img, self.labels[idx]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if x.ndim > 2:
            x = x.view(x.shape[0], -1)
        h = self.feature(x)
        logits = self.classifier(h)
        if return_features:
            return logits, h
        return logits


def add_spurious_feature(x: np.ndarray, y: np.ndarray, p: float, rng: np.random.RandomState) -> np.ndarray:
    spurious = y.copy()
    flips = rng.rand(y.shape[0]) > p
    spurious[flips] = 1 - spurious[flips]
    spurious = spurious.astype(np.float32).reshape(-1, 1)
    return np.concatenate([x, spurious], axis=1)


def build_colored_mnist_envs(data_root: str, seed: int, max_train: int | None, max_test: int | None):
    train = datasets.MNIST(root=data_root, train=True, download=True)
    test = datasets.MNIST(root=data_root, train=False, download=True)
    train_images = train.data.float() / 255.0
    test_images = test.data.float() / 255.0
    train_labels = (train.targets < 5).long()
    test_labels = (test.targets < 5).long()

    rng = np.random.RandomState(seed)
    if max_train is not None and max_train < len(train_labels):
        indices = rng.choice(len(train_labels), size=max_train, replace=False)
    else:
        indices = rng.permutation(len(train_labels))
    splits = np.array_split(indices, 3)
    train_envs = []
    probs = [0.9, 0.8, 0.7]
    for env_id, split in enumerate(splits):
        env_images, env_labels = train_images[split], train_labels[split]
        colored = colorize_images(env_images, env_labels, probs[env_id], seed + env_id)
        train_envs.append(TensorDataset(colored, env_labels))

    if max_test is not None and max_test < len(test_labels):
        test_indices = rng.choice(len(test_labels), size=max_test, replace=False)
        test_images = test_images[test_indices]
        test_labels = test_labels[test_indices]
    test_images_colored = colorize_images(test_images, test_labels, 0.1, seed + 10)
    test_envs = [TensorDataset(test_images_colored, test_labels)]
    input_shape = (3, 28, 28)
    return train_envs, test_envs, input_shape, 2


def colorize_images(images: torch.Tensor, labels: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    color = labels.clone()
    flips = torch.rand(labels.shape[0], generator=rng) > p
    color[flips] = 1 - color[flips]
    color = color.view(-1, 1, 1)
    colored = images.unsqueeze(1).repeat(1, 3, 1, 1)
    red_mask = (color == 0).float()
    green_mask = (color == 1).float()
    colored[:, 0] = colored[:, 0] * red_mask
    colored[:, 1] = colored[:, 1] * green_mask
    colored[:, 2] = 0.0
    return colored


def build_rotated_mnist_envs(data_root: str, seed: int, max_train: int | None, max_test: int | None):
    train = datasets.MNIST(root=data_root, train=True, download=True)
    test = datasets.MNIST(root=data_root, train=False, download=True)
    train_images = train.data
    test_images = test.data
    train_labels = train.targets
    test_labels = test.targets

    rng = np.random.RandomState(seed)
    if max_train is not None and max_train < len(train_labels):
        indices = rng.choice(len(train_labels), size=max_train, replace=False)
    else:
        indices = rng.permutation(len(train_labels))
    splits = np.array_split(indices, 3)
    train_angles = [0.0, 15.0, 30.0]
    train_envs = []
    for env_id, split in enumerate(splits):
        train_envs.append(RotatedMNISTSubset(train_images, train_labels, split, train_angles[env_id]))

    test_angles = [45.0, 60.0]
    test_envs = []
    if max_test is not None and max_test < len(test_labels):
        test_indices = rng.choice(len(test_labels), size=max_test, replace=False)
    else:
        test_indices = np.arange(len(test_labels))
    for angle in test_angles:
        test_envs.append(RotatedMNISTSubset(test_images, test_labels, test_indices, angle))

    input_shape = (1, 28, 28)
    return train_envs, test_envs, input_shape, 10


def build_breast_cancer_envs(seed: int):
    data = load_breast_cancer()
    x = data.data
    y = data.target
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=seed, stratify=y
    )

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(y_train))
    splits = np.array_split(indices, 3)
    probs = [0.9, 0.8, 0.7]
    train_envs = []
    for env_id, split in enumerate(splits):
        env_x = x_train[split]
        env_y = y_train[split]
        env_x = add_spurious_feature(env_x, env_y, probs[env_id], np.random.RandomState(seed + env_id))
        train_envs.append(
            TensorDataset(torch.tensor(env_x, dtype=torch.float32), torch.tensor(env_y, dtype=torch.long))
        )

    test_x = add_spurious_feature(x_test, y_test, 0.1, np.random.RandomState(seed + 10))
    test_envs = [
        TensorDataset(torch.tensor(test_x, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    ]
    input_shape = (test_x.shape[1],)
    return train_envs, test_envs, input_shape, 2


def build_synthetic_envs(seed: int, n_samples: int = 60000):
    x, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        class_sep=1.0,
        flip_y=0.02,
        random_state=seed,
    )
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=seed, stratify=y
    )
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(y_train))
    splits = np.array_split(indices, 3)
    probs = [0.95, 0.85, 0.7]
    train_envs = []
    for env_id, split in enumerate(splits):
        env_x = x_train[split]
        env_y = y_train[split]
        env_x = add_spurious_feature(env_x, env_y, probs[env_id], np.random.RandomState(seed + 100 + env_id))
        train_envs.append(
            TensorDataset(torch.tensor(env_x, dtype=torch.float32), torch.tensor(env_y, dtype=torch.long))
        )

    test_x = add_spurious_feature(x_test, y_test, 0.1, np.random.RandomState(seed + 110))
    test_envs = [
        TensorDataset(torch.tensor(test_x, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    ]
    input_shape = (test_x.shape[1],)
    return train_envs, test_envs, input_shape, 2


def compute_label_conditional_penalty(features_list, labels_list, num_classes: int) -> torch.Tensor:
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


def compute_label_conditional_coral_penalty(features_list, labels_list, num_classes: int) -> torch.Tensor:
    penalty = 0.0
    for cls in range(num_classes):
        env_covs = []
        for feats, labels in zip(features_list, labels_list):
            mask = labels == cls
            if mask.sum() < 3:
                continue
            centered = feats[mask] - feats[mask].mean(dim=0, keepdim=True)
            cov = centered.t().mm(centered) / max(centered.shape[0] - 1, 1)
            env_covs.append(cov)
        if len(env_covs) <= 1:
            continue
        cov_mean = sum(env_covs) / len(env_covs)
        penalty = penalty + sum(((cov - cov_mean) ** 2).mean() for cov in env_covs)
    return penalty


def compute_coral_penalty(features_list) -> torch.Tensor:
    covs = []
    for feats in features_list:
        feats = feats - feats.mean(dim=0, keepdim=True)
        cov = feats.t().mm(feats) / max(feats.shape[0] - 1, 1)
        covs.append(cov)
    cov_mean = sum(covs) / len(covs)
    penalty = sum(((cov - cov_mean) ** 2).mean() for cov in covs)
    return penalty


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def build_loaders(envs, batch_size: int, shuffle: bool):
    return [
        DataLoader(env, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)
        for env in envs
    ]


def train_epoch(
    model,
    env_loaders,
    optimizer,
    criterion,
    method: str,
    method_hparams: dict,
    num_classes: int,
    device: torch.device,
    env_weights,
):
    model.train()
    total_loss = 0.0
    entropies = []
    weight_maxes = []
    weight_mins = []
    batches = zip(*env_loaders)
    for batch in batches:
        env_losses = []
        features_list = []
        labels_list = []
        logits_list = []
        for env_x, env_y in batch:
            env_x = env_x.to(device)
            env_y = env_y.to(device)
            logits, feats = model(env_x, return_features=True)
            loss = criterion(logits, env_y)
            env_losses.append(loss)
            features_list.append(feats)
            labels_list.append(env_y)
            logits_list.append(logits)

        losses = torch.stack(env_losses)
        if method == "erm":
            loss = losses.mean()
            weights = torch.ones_like(losses) / losses.numel()
        elif method == "groupdro":
            with torch.no_grad():
                env_weights = env_weights * torch.exp(method_hparams["gdro_eta"] * losses.detach())
                env_weights = env_weights / env_weights.sum()
            loss = (env_weights * losses).sum()
            weights = env_weights
        elif method == "rex":
            loss = losses.mean() + method_hparams["rex_lambda"] * losses.var(unbiased=False)
            weights = torch.ones_like(losses) / losses.numel()
        elif method == "irm":
            scale = torch.tensor(1.0, device=device, requires_grad=True)
            irm_penalty = 0.0
            for logits, labels in zip(logits_list, labels_list):
                irm_loss = criterion(logits * scale, labels)
                grad = torch.autograd.grad(irm_loss, [scale], create_graph=True)[0]
                irm_penalty = irm_penalty + grad.pow(2)
            loss = losses.mean() + method_hparams["irm_lambda"] * irm_penalty
            weights = torch.ones_like(losses) / losses.numel()
        elif method == "coral":
            coral_penalty = compute_coral_penalty(features_list)
            loss = losses.mean() + method_hparams["coral_lambda"] * coral_penalty
            weights = torch.ones_like(losses) / losses.numel()
        elif method in {"ipdr", "ipdr_inv", "ipdr_no_ent", "ipdr_no_inv", "ipdr_dro_inv", "ipdr_inv_cov"}:
            use_hard_max = method == "ipdr_no_ent"
            use_dro = method in {"ipdr", "ipdr_no_inv", "ipdr_dro_inv"}
            use_ent_smoothing = method in {"ipdr", "ipdr_no_inv"}

            use_inv_mean = method in {"ipdr", "ipdr_inv", "ipdr_no_ent", "ipdr_dro_inv", "ipdr_inv_cov"}
            use_inv_cov = method == "ipdr_inv_cov"

            if use_hard_max:
                max_idx = torch.argmax(losses)
                weights = torch.zeros_like(losses)
                weights[max_idx] = 1.0
                base_loss = (weights * losses).sum()
            elif use_dro:
                with torch.no_grad():
                    env_weights = env_weights * torch.exp(method_hparams["gdro_eta"] * losses.detach())
                    env_weights = env_weights / env_weights.sum()
                    if use_ent_smoothing:
                        mix = float(method_hparams["ent_lambda"])
                        uniform = torch.ones_like(env_weights) / env_weights.numel()
                        env_weights = (1.0 - mix) * env_weights + mix * uniform
                weights = env_weights
                base_loss = (weights * losses).sum()
            else:
                weights = torch.ones_like(losses) / losses.numel()
                base_loss = losses.mean()

            if use_inv_mean:
                inv_penalty = compute_label_conditional_penalty(features_list, labels_list, num_classes)
                base_loss = base_loss + method_hparams["inv_lambda"] * inv_penalty
            if use_inv_cov:
                inv_cov_penalty = compute_label_conditional_coral_penalty(
                    features_list, labels_list, num_classes
                )
                base_loss = base_loss + method_hparams["inv_cov_lambda"] * inv_cov_penalty
            loss = base_loss
        else:
            raise ValueError(f"Unknown method: {method}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        entropy = float((-(weights * torch.log(weights + 1e-8)).sum()).item())
        entropies.append(entropy)
        weight_maxes.append(float(weights.max().item()))
        weight_mins.append(float(weights.min().item()))

    diagnostics = {
        "entropy": float(np.mean(entropies)) if entropies else float("nan"),
        "weight_max": float(np.mean(weight_maxes)) if weight_maxes else float("nan"),
        "weight_min": float(np.mean(weight_mins)) if weight_mins else float("nan"),
    }
    return total_loss / max(1, len(env_loaders[0])), env_weights, diagnostics


@torch.no_grad()
def evaluate(model, env_loaders, criterion, device: torch.device):
    model.eval()
    env_accs = []
    env_losses = []
    for loader in env_loaders:
        total = 0
        correct = 0
        total_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.shape[0]
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.shape[0]
        env_accs.append(correct / max(1, total))
        env_losses.append(total_loss / max(1, total))
    env_accs = np.array(env_accs)
    env_losses = np.array(env_losses)
    return env_accs, env_losses, env_accs.mean(), env_accs.min()


def run_experiment(dataset: str, method: str, seed: int, device: torch.device, args) -> dict:
    set_seed(seed)
    if dataset in {"colored_mnist", "rotated_mnist"}:
        data_root = os.path.join(args.data_root, "mnist")
    else:
        data_root = os.path.join(args.data_root, dataset)
    os.makedirs(data_root, exist_ok=True)

    if dataset == "colored_mnist":
        train_envs, test_envs, input_shape, num_classes = build_colored_mnist_envs(
            data_root, seed, args.max_mnist_train, args.max_mnist_test
        )
        batch_size = 256
        epochs = 5
        hidden_dim = 256
        feature_dim = 128
    elif dataset == "rotated_mnist":
        train_envs, test_envs, input_shape, num_classes = build_rotated_mnist_envs(
            data_root, seed, args.max_mnist_train, args.max_mnist_test
        )
        batch_size = 256
        epochs = 5
        hidden_dim = 256
        feature_dim = 128
    elif dataset == "breast_cancer":
        train_envs, test_envs, input_shape, num_classes = build_breast_cancer_envs(seed)
        batch_size = 64
        epochs = 50
        hidden_dim = 128
        feature_dim = 64
    elif dataset == "synthetic_spurious":
        train_envs, test_envs, input_shape, num_classes = build_synthetic_envs(seed)
        batch_size = 512
        epochs = 20
        hidden_dim = 128
        feature_dim = 64
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if args.quick:
        epochs = max(2, epochs // 2)

    input_dim = int(np.prod(input_shape))
    model = MLP(input_dim, hidden_dim, feature_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    method_hparams = {
        "gdro_eta": 0.1,
        "rex_lambda": 1.0,
        "irm_lambda": 1.0,
        "coral_lambda": 1.0,
        "inv_lambda": 1.0,
        "inv_cov_lambda": 1.0,
        "ent_lambda": 0.1,
    }

    env_weights = torch.ones(len(train_envs), device=device) / len(train_envs)
    train_loaders = build_loaders(train_envs, batch_size=batch_size, shuffle=True)
    test_loaders = build_loaders(test_envs, batch_size=batch_size, shuffle=False)

    weight_entropy_curve = []
    weight_max_curve = []
    weight_min_curve = []
    for epoch in range(epochs):
        _, env_weights, diagnostics = train_epoch(
            model,
            train_loaders,
            optimizer,
            criterion,
            method,
            method_hparams,
            num_classes,
            device,
            env_weights,
        )
        weight_entropy_curve.append(diagnostics["entropy"])
        weight_max_curve.append(diagnostics["weight_max"])
        weight_min_curve.append(diagnostics["weight_min"])

    env_accs, env_losses, mean_acc, worst_acc = evaluate(model, test_loaders, criterion, device)
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "mean_acc": float(mean_acc),
        "worst_acc": float(worst_acc),
        "test_env_accs": env_accs.tolist(),
        "test_env_losses": env_losses.tolist(),
        "num_classes": num_classes,
        "epochs": epochs,
        "weight_entropy_curve": json.dumps(weight_entropy_curve),
        "weight_max_curve": json.dumps(weight_max_curve),
        "weight_min_curve": json.dumps(weight_min_curve),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["colored_mnist", "rotated_mnist", "breast_cancer", "synthetic_spurious"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "erm",
            "groupdro",
            "rex",
            "irm",
            "coral",
            "ipdr",
            "ipdr_inv",
            "ipdr_no_ent",
            "ipdr_no_inv",
            "ipdr_dro_inv",
            "ipdr_inv_cov",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--data-root", default=os.path.join("exp", "data"))
    parser.add_argument("--results-path", default=os.path.join("exp", "results", "results.csv"))
    parser.add_argument("--max-mnist-train", type=int, default=20000)
    parser.add_argument("--max-mnist-test", type=int, default=5000)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    start = time.time()
    for dataset in args.datasets:
        for method in args.methods:
            for seed in args.seeds:
                result = run_experiment(dataset, method, seed, device, args)
                records.append(result)
                print(
                    f"[{dataset}][{method}][seed={seed}] mean={result['mean_acc']:.4f} "
                    f"worst={result['worst_acc']:.4f}"
                )

    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(args.results_path, index=False)

    meta_path = os.path.join(os.path.dirname(args.results_path), "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "datasets": args.datasets,
                "methods": args.methods,
                "seeds": args.seeds,
                "device": str(device),
                "elapsed_sec": time.time() - start,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
