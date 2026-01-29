import os
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

# ======================
# CONFIG
# ======================
TRAIN_DIR = "data/train"
VAL_DIR   = "data/val"

IMG_SIZE = 224
BATCH = 32
EPOCHS = 15
LR = 3e-4

# Windows tip: if dataloader crashes, set this to 0
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "best_model.pth"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# ======================
# TRANSFORMS
# ======================
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ======================
# METRICS
# ======================
def evaluate(model, loader, num_classes: int):
    model.eval()
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            for yi, pi in zip(y.tolist(), preds.tolist()):
                per_class_total[yi] += 1
                if yi == pi:
                    per_class_correct[yi] += 1

    overall_acc = correct / max(1, total)

    per_class_acc = {}
    for c in range(num_classes):
        if per_class_total[c] == 0:
            per_class_acc[c] = None
        else:
            per_class_acc[c] = per_class_correct[c] / per_class_total[c]

    return overall_acc, per_class_acc

# ======================
# MODEL LOADER (handles old torchvision)
# ======================
def build_model(num_classes: int):
    # New torchvision
    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    except Exception:
        # Old torchvision fallback
        model = models.efficientnet_b0(pretrained=True)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

# ======================
# SANITY CHECK: folder exists + has classes
# ======================
def check_dir(path: str, name: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{name} folder not found: {path}")

    classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if not classes:
        raise ValueError(f"{name} folder has no class subfolders: {path}")

    # also check images exist somewhere
    found_img = False
    for c in classes:
        cp = os.path.join(path, c)
        for f in os.listdir(cp):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                found_img = True
                break
        if found_img:
            break

    if not found_img:
        raise ValueError(f"{name} folder has class subfolders but no images: {path}")

def main():
    # Make sure folders exist
    check_dir(TRAIN_DIR, "TRAIN_DIR")
    check_dir(VAL_DIR, "VAL_DIR")

    # Datasets
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
    val_ds   = datasets.ImageFolder(VAL_DIR, transform=val_tfms)

    class_names = train_ds.classes
    num_classes = len(class_names)

    # Ensure val has same classes (or at least same mapping)
    if val_ds.class_to_idx != train_ds.class_to_idx:
        raise ValueError(
            "Class mapping mismatch between train and val.\n"
            f"Train classes: {train_ds.class_to_idx}\n"
            f"Val classes:   {val_ds.class_to_idx}\n"
            "Fix by ensuring val has same class folder names as train."
        )

    print("[INFO] Classes:", class_names)
    print("[INFO] Num classes:", num_classes)
    print("[INFO] Device:", DEVICE)

    # Weighted Sampler
    targets = [label for _, label in train_ds.samples]
    counts = Counter(targets)

    # Ensure every class index exists in counts (avoid KeyError later)
    for c in range(num_classes):
        if c not in counts:
            counts[c] = 1  # tiny safety net, but ideally your train has all classes

    class_weights_for_sampler = {
        cls: len(targets) / (num_classes * ccount)
        for cls, ccount in counts.items()
    }

    sample_weights = torch.DoubleTensor([class_weights_for_sampler[t] for t in targets])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda")
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda")
    )

    # Model
    model = build_model(num_classes).to(DEVICE)

    # Loss weights (handles missing classes safely)
    class_counts = torch.tensor([counts[i] for i in range(num_classes)], dtype=torch.float)
    loss_weights = (class_counts.sum() / (num_classes * class_counts)).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            seen += y.size(0)

        train_loss = running_loss / max(1, seen)
        val_acc, per_class = evaluate(model, val_loader, num_classes)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val overall accuracy: {val_acc:.4f}")
        for idx, cls_name in enumerate(class_names):
            v = per_class[idx]
            print(f"  {cls_name:6s}: {'N/A' if v is None else f'{v:.4f}'}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "img_size": IMG_SIZE
                },
                SAVE_PATH
            )
            print(f"[SAVE] Best model saved to {SAVE_PATH} (val_acc={best_val_acc:.4f})")

    print(f"\n[DONE] Best val accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
