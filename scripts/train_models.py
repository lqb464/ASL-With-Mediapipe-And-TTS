import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

from src.models.model import SequenceRNNClassifier, SequenceRNNConfig


try:
    import wandb
except ImportError:
    wandb = None


with open("configs/train.yaml", encoding="utf-8") as f:
    TRAIN_CFG = yaml.safe_load(f)["train"]


class ASLSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = int(self.y[idx])

        nonzero_frames = np.any(x != 0.0, axis=1)
        length = int(nonzero_frames.sum())
        length = max(length, 1)

        return (
            torch.from_numpy(x).float(),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )


def load_label_map(base_path: Path, npz_data=None):
    json_path = base_path.with_suffix(".labels.json")

    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    if npz_data is not None and "label_map" in npz_data:
        return json.loads(str(npz_data["label_map"]))

    raise FileNotFoundError("Không tìm thấy label_map trong .npz hoặc .labels.json")


def load_dataset(path: Path):
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        label_map = load_label_map(path, data)
        return X, y, label_map

    if path.suffix == ".npy":
        X_path = path
        base = Path(str(path).replace(".X.npy", ""))
    else:
        base = path.with_suffix("")
        X_path = base.with_suffix(".X.npy")

    y_path = base.with_suffix(".y.npy")

    if not X_path.exists():
        raise FileNotFoundError(f"Không tìm thấy X file: {X_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Không tìm thấy y file: {y_path}")

    X = np.load(X_path, mmap_mode="r")
    y = np.load(y_path)
    label_map = load_label_map(base)

    return X, y, label_map


def init_wandb(enabled, project, run_name, config):
    if not enabled:
        return None

    if wandb is None:
        print("[!] wandb chưa được cài. Chạy: pip install wandb")
        return None

    return wandb.init(
        project=project,
        name=run_name,
        config=config,
    )


def train(
    data_path: str | Path = "data/processed/train.npz",
    use_wandb: bool = False,
    wandb_project: str = "asl-recognizer",
    wandb_run_name: str | None = None,
):
    seed = int(TRAIN_CFG.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_path = Path(data_path)

    if not data_path.exists():
        alt_path = Path("data/processed/train.X.npy")
        if alt_path.exists():
            data_path = alt_path
        else:
            print(f"Không tìm thấy dataset: {data_path}")
            print("Hãy build data trước bằng: python -m scripts.build_dataset --source raw")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, label_map = load_dataset(data_path)

    input_dim = int(X.shape[2])
    num_classes = len(label_map)

    dataset = ASLSequenceDataset(X, y)

    val_split = float(TRAIN_CFG.get("val_split", 0.2))
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    batch_size = int(TRAIN_CFG.get("batch_size", 32))
    epochs = int(TRAIN_CFG.get("epochs", 20))
    lr = float(TRAIN_CFG.get("lr", 1e-3))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    config = SequenceRNNConfig(input_dim=input_dim, num_classes=num_classes)
    model = SequenceRNNClassifier(config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    save_path = Path("models/checkpoints/best_model.pt")
    best_acc = 0.0

    wandb_run = init_wandb(
        enabled=use_wandb,
        project=wandb_project,
        run_name=wandb_run_name,
        config={
            "data_path": str(data_path),
            "x_shape": tuple(X.shape),
            "input_dim": input_dim,
            "num_classes": num_classes,
            "train_size": train_size,
            "val_size": val_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "seed": seed,
            "device": str(device),
            "model_type": config.model_type,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "bidirectional": config.bidirectional,
        },
    )

    if wandb_run is not None:
        wandb.watch(model, log="gradients", log_freq=100)

    print("=" * 60)
    print("BẮT ĐẦU TRAINING")
    print("=" * 60)
    print(f"Device       : {device}")
    print(f"Dataset      : {data_path}")
    print(f"X shape      : {X.shape}")
    print(f"Input dim    : {input_dim}")
    print(f"Classes      : {num_classes}")
    print(f"Train / Val  : {train_size} / {val_size}")
    print(f"Batch size   : {batch_size}")
    print(f"Epochs       : {epochs}")
    print(f"WandB        : {'ON' if wandb_run is not None else 'OFF'}")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_train = 0
        correct_train = 0

        for batch_X, batch_y, lengths in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X, lengths)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)

        train_acc = correct_train / max(total_train, 1)

        model.eval()
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_X, batch_y, lengths in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                lengths = lengths.to(device)

                outputs = model(batch_X, lengths)
                preds = outputs.argmax(dim=1)

                correct_val += (preds == batch_y).sum().item()
                total_val += batch_y.size(0)

        val_acc = correct_val / max(total_val, 1)
        avg_loss = total_loss / max(len(train_loader), 1)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if wandb_run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_loss,
                "train/acc": train_acc,
                "val/acc": val_acc,
                "best_val_acc": best_acc,
            })

        if val_acc > best_acc:
            best_acc = val_acc
            model.save(
                save_path,
                extra={
                    "label_map": label_map,
                    "best_val_acc": best_acc,
                    "input_shape": tuple(X.shape),
                },
            )
            print(f"  [✓] Saved best model: {save_path} | Val Acc: {best_acc:.4f}")

            if wandb_run is not None:
                wandb.run.summary["best_val_acc"] = best_acc
                wandb.save(str(save_path))

    print("=" * 60)
    print(f"TRAINING DONE. Best Val Acc: {best_acc:.4f}")
    print("=" * 60)

    if wandb_run is not None:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/train.npz"),
        help="Path tới train.npz hoặc train.X.npy",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Bật logging với Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="asl-recognizer",
        help="Tên project trên wandb.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Tên run trên wandb.",
    )
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()