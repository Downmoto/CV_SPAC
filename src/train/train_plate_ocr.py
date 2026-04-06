"""train the crnn plate recognition model on extracted plate crops."""

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.ocr.plate_crnn import (
    NUM_CLASSES,
    PlateRecCRNN,
    encode_text,
    decode_output,
)


IMG_H = 32
IMG_W = 128


class PlateDataset(Dataset):
    def __init__(
        self,
        crops_dir: Path,
        labels: list[tuple[str, str]],
        img_h: int = IMG_H,
        img_w: int = IMG_W,
        augment: bool = False,
    ) -> None:
        self.crops_dir = crops_dir
        self.labels = labels
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]

        # random brightness/contrast
        if random.random() < 0.5:
            alpha = random.uniform(0.7, 1.3)
            beta = random.randint(-30, 30)
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # random gaussian noise
        if random.random() < 0.4:
            noise = np.random.normal(0, random.uniform(5, 25), img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # random gaussian blur
        if random.random() < 0.3:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        # random rotation
        if random.random() < 0.4:
            angle = random.uniform(-8, 8)
            center = (w // 2, h // 2)
            rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rot, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # random perspective warp
        if random.random() < 0.3:
            margin = int(min(h, w) * 0.08)
            src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            dst = np.array([
                [random.randint(0, margin), random.randint(0, margin)],
                [w - random.randint(0, margin), random.randint(0, margin)],
                [w - random.randint(0, margin), h - random.randint(0, margin)],
                [random.randint(0, margin), h - random.randint(0, margin)],
            ], dtype=np.float32)
            mat = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(img, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # random erosion/dilation
        if random.random() < 0.2:
            kernel = np.ones((2, 2), np.uint8)
            if random.random() < 0.5:
                img = cv2.erode(img, kernel, iterations=1)
            else:
                img = cv2.dilate(img, kernel, iterations=1)

        return img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[int], int]:
        filename, label = self.labels[idx]
        img_path = self.crops_dir / filename

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # fallback: blank image
            img = np.zeros((self.img_h, self.img_w), dtype=np.uint8)

        if self.augment:
            img = self._augment(img)

        # resize keeping aspect ratio, pad to fixed width
        h, w = img.shape[:2]
        ratio = self.img_h / max(1, h)
        new_w = min(int(w * ratio), self.img_w)
        img = cv2.resize(img, (new_w, self.img_h), interpolation=cv2.INTER_CUBIC)

        # pad to img_w
        if new_w < self.img_w:
            pad = np.zeros((self.img_h, self.img_w - new_w), dtype=np.uint8)
            img = np.concatenate([img, pad], axis=1)

        # normalize to [0, 1]
        tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        target = encode_text(label)
        return tensor, target, len(target)


def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, dim=0)

    # flatten targets
    flat_targets = []
    for t in targets:
        flat_targets.extend(t)

    return (
        images,
        torch.IntTensor(flat_targets),
        torch.IntTensor(target_lengths),
    )


def load_labels(labels_csv: Path, splits: list[str]) -> list[tuple[str, str]]:
    labels = []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] in splits:
                labels.append((row["filename"], row["label"]))
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train crnn plate recognition model")
    parser.add_argument("--crops-dir", default="data/plate_crops/images")
    parser.add_argument("--labels-csv", default="data/plate_crops/labels.csv")
    parser.add_argument("--output-weights", default="models/plate_crnn.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--repeat-factor", type=int, default=10,
                        help="repeat dataset n times per epoch for more augmentation")
    return parser.parse_args()


class RepeatedDataset(Dataset):
    """repeat a dataset multiple times to increase effective epoch size with augmentation."""

    def __init__(self, base: PlateDataset, factor: int) -> None:
        self.base = base
        self.factor = factor

    def __len__(self) -> int:
        return len(self.base) * self.factor

    def __getitem__(self, idx: int):
        return self.base[idx % len(self.base)]


def evaluate(model, dataloader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets, target_lengths in dataloader:
            images = images.to(device)
            output = model(images)
            seq_len = output.size(0)
            batch_size = images.size(0)
            input_lengths = torch.IntTensor([seq_len] * batch_size)

            loss = criterion(output, targets, input_lengths, target_lengths)
            total_loss += loss.item()

            # greedy decode
            _, preds = output.max(2)
            preds = preds.transpose(0, 1).cpu().numpy()
            offset = 0
            for i in range(batch_size):
                pred_text = decode_output(preds[i].tolist())
                tlen = target_lengths[i].item()
                gt_indices = targets[offset : offset + tlen].tolist()
                gt_text = decode_output(gt_indices)
                if pred_text == gt_text:
                    correct += 1
                total += 1
                offset += tlen

    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    crops_dir = Path(args.crops_dir)
    labels_csv = Path(args.labels_csv)

    if not labels_csv.exists():
        raise FileNotFoundError(
            f"labels csv not found: {labels_csv}. "
            "run extract_plate_crops first."
        )

    # load labels
    train_labels = load_labels(labels_csv, ["train", "val"])
    test_labels = load_labels(labels_csv, ["test"])

    if not train_labels:
        raise ValueError("no training samples found in labels csv")

    print(f"training samples: {len(train_labels)}")
    print(f"test samples: {len(test_labels)}")

    # datasets
    train_ds = PlateDataset(crops_dir, train_labels, augment=True)
    repeated_ds = RepeatedDataset(train_ds, args.repeat_factor)
    train_loader = DataLoader(
        repeated_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    test_ds = PlateDataset(crops_dir, test_labels, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # model
    model = PlateRecCRNN(
        img_h=IMG_H,
        num_classes=NUM_CLASSES,
        hidden_size=args.hidden_size,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {param_count:,}")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    output_path = Path(args.output_weights)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for images, targets, target_lengths in train_loader:
            images = images.to(device)
            output = model(images)  # (seq_len, batch, classes)

            seq_len = output.size(0)
            batch_size = images.size(0)
            input_lengths = torch.IntTensor([seq_len] * batch_size)

            loss = criterion(output, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(1, batch_count)

        # evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs:
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"epoch {epoch:4d}/{args.epochs}  "
                f"train_loss={avg_train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.4f}  "
                f"lr={lr:.6f}"
            )

            if val_acc >= best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), output_path)
                print(f"  -> saved best model (acc={best_acc:.4f})")

    print(f"\ntraining complete. best val accuracy: {best_acc:.4f}")
    print(f"model saved to: {output_path}")


if __name__ == "__main__":
    main()
