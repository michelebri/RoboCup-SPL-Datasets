#!/usr/bin/env python3
"""Train the robot color classifier CNN from train/valid CSV splits."""

import argparse
import csv
import pathlib

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import yaml

from color_classifier import ColorClassifierCNN
from extract_robot_pics import COLOR_MAP


class RobotColorDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, label = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_color_index() -> dict:
    ordered_colors = [COLOR_MAP[key] for key in sorted(COLOR_MAP.keys())]
    return {color: idx for idx, color in enumerate(ordered_colors)}


def load_split(csv_path: pathlib.Path, base_dir: pathlib.Path, color_to_idx: dict) -> list:
    items = []
    with csv_path.open(mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_rel = (row.get("image") or "").strip()
            color = (row.get("color") or "").strip()
            if not image_rel or not color:
                continue
            if color not in color_to_idx:
                continue
            img_path = base_dir / image_rel
            if not img_path.is_file():
                candidate = base_dir / "robot_pics" / "images" / image_rel
                if candidate.is_file():
                    img_path = candidate
            items.append((img_path, color_to_idx[color]))
    return items


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train color classifier CNN from CSV splits")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("train_color_classifier.yml"))
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = pathlib.Path(__file__).resolve().parent / config_path
    with config_path.open(mode="r", encoding="utf-8", newline="\n") as f:
        config = yaml.safe_load(f) or {}
    base_dir = config_path.resolve().parent

    csv_root = pathlib.Path(config.get("csv_root", "color_dataset"))
    if not csv_root.is_absolute():
        csv_root = base_dir / csv_root
    epochs = int(config.get("epochs", 15))
    batch_size = int(config.get("batch_size", 64))
    lr = float(config.get("lr", 1e-3))
    workers = int(config.get("workers", 2))
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output = pathlib.Path(config.get("output", "color_classifier.pt"))
    if not output.is_absolute():
        output = base_dir / output

    train_csv = csv_root / "train.csv"
    valid_csv = csv_root / "valid.csv"
    if not train_csv.is_file():
        raise SystemExit("Missing train.csv in csv_root. Use the split_dataset.py script to create train/valid/test splits from the extracted robot pics.")
    if not valid_csv.is_file():
        raise SystemExit("Missing valid.csv in csv_root. Use the split_dataset.py script to create train/valid/test splits from the extracted robot pics.")

    color_to_idx = build_color_index()
    train_items = load_split(train_csv, csv_root.parent, color_to_idx)
    valid_items = load_split(valid_csv, csv_root.parent, color_to_idx)
    classes = [COLOR_MAP[key] for key in sorted(COLOR_MAP.keys())]

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_loader = DataLoader(
        RobotColorDataset(train_items, transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )
    val_loader = DataLoader(
        RobotColorDataset(valid_items, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    model = ColorClassifierCNN(in_channels=3, num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                },
                output,
            )

    print(f"Saved best model to {output}")


if __name__ == "__main__":
    main()
