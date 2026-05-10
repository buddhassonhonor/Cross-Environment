import argparse
import os
import random

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


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
    weights: torch.Tensor, losses: torch.Tensor, eta: float, ent_lambda: float
) -> torch.Tensor:
    new_weights = weights * torch.exp(eta * losses.detach())
    new_weights = new_weights / new_weights.sum()
    uniform = torch.ones_like(new_weights) / len(new_weights)
    new_weights = (1.0 - ent_lambda) * new_weights + ent_lambda * uniform
    return new_weights


def parse_args():
    parser = argparse.ArgumentParser(description="t-SNE for linear probing models")
    parser.add_argument(
        "--features-dir", default=os.path.join("exp", "linear_probe", "features", "PACS")
    )
    parser.add_argument("--output", default=os.path.join("exp", "linear_probe", "tsne_pacs.png"))
    parser.add_argument("--test-env", default="sketch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--inv-lambda", type=float, default=1.0)
    parser.add_argument("--ent-lambda", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=2000)
    return parser.parse_args()


def load_features(features_dir: str):
    envs = []
    features = []
    labels = []
    for name in sorted(os.listdir(features_dir)):
        if not name.endswith(".pt"):
            continue
        env = name.replace(".pt", "")
        payload = torch.load(os.path.join(features_dir, name))
        envs.append(env)
        features.append(payload["features"])
        labels.append(payload["labels"])
    return envs, features, labels


def train_model(
    features_by_env,
    labels_by_env,
    test_env_idx: int,
    method: str,
    args,
    device: torch.device,
):
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

    return model


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs, features_by_env, labels_by_env = load_features(args.features_dir)
    if args.test_env not in envs:
        raise ValueError(f"Unknown test env: {args.test_env}")
    test_env_idx = envs.index(args.test_env)

    model_erm = train_model(
        features_by_env, labels_by_env, test_env_idx, "erm", args, device
    )
    model_ipdr = train_model(
        features_by_env, labels_by_env, test_env_idx, "ipdr", args, device
    )

    test_feats = features_by_env[test_env_idx]
    test_labels = labels_by_env[test_env_idx]
    if args.max_samples and len(test_labels) > args.max_samples:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(test_labels), size=args.max_samples, replace=False)
        test_feats = test_feats[idx]
        test_labels = test_labels[idx]

    with torch.no_grad():
        _, feats_erm = model_erm(test_feats.to(device), return_features=True)
        _, feats_ipdr = model_ipdr(test_feats.to(device), return_features=True)
    feats_erm = feats_erm.cpu().numpy()
    feats_ipdr = feats_ipdr.cpu().numpy()
    labels = test_labels.numpy()

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(n_components=2, init="pca", random_state=args.seed, perplexity=30)
    emb_erm = tsne.fit_transform(feats_erm)
    tsne = TSNE(n_components=2, init="pca", random_state=args.seed, perplexity=30)
    emb_ipdr = tsne.fit_transform(feats_ipdr)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=220)
    for ax, emb, title in zip(
        axes, [emb_erm, emb_ipdr], ["ERM", "IPDR"]
    ):
        scatter = ax.scatter(
            emb[:, 0],
            emb[:, 1],
            c=labels,
            s=13,
            cmap="tab10",
            alpha=0.78,
        )
        ax.set_title(f"{title} ({args.test_env})")
        ax.set_xlabel("t-SNE-1")
        ax.set_ylabel("t-SNE-2")
        ax.set_xticks([])
        ax.set_yticks([])
    handles, _ = scatter.legend_elements(num=7)
    fig.legend(
        handles,
        [str(i) for i in range(7)],
        title="Class",
        loc="center right",
        title_fontsize=12,
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
