import sys
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =====================
# CONFIG
# =====================
DATA_DIR = Path("../data/val")  # validation folder
WEIGHTS = Path("../runs/train-cls/skin_lesion_cls_gpu/weights/best.pt")
IMG_SIZE = 224
BATCH_SIZE = 64
DEVICE = "cuda:0"  # HARD GPU


def _ensure_yolov5_in_path() -> None:
    """
    YOLOv5 saves models that reference a top-level 'models' package.
    When we run this script from 'backend/', we must add the YOLOv5
    repo root (which contains the 'models' package) to sys.path so
    that torch.load can unpickle the checkpoint.
    """
    repo_root = Path(__file__).resolve().parents[1]  # project root
    yolov5_dir = repo_root / "yolov5"

    if yolov5_dir.is_dir():
        yolov5_str = str(yolov5_dir)
        if yolov5_str not in sys.path:
            sys.path.insert(0, yolov5_str)
        # Import 'models' so that it's available during unpickling
        try:
            __import__("models")
        except ImportError:
            # If this fails, torch.load will still raise a clear error.
            pass


def main():
    assert torch.cuda.is_available(), "CUDA not available. Fix your torch install."
    device = torch.device(DEVICE)

    # Make sure YOLOv5 'models' package is importable for torch.load
    _ensure_yolov5_in_path()

    print("[INFO] Loading model...")
    model = torch.load(WEIGHTS, map_location=device)["model"].float().eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    correct = 0
    total = 0
    class_correct = {cls: 0 for cls in dataset.classes}
    class_total = {cls: 0 for cls in dataset.classes}

    print("[INFO] Evaluating...")
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for i in range(len(labels)):
                cls = dataset.classes[labels[i]]
                class_total[cls] += 1
                if preds[i] == labels[i]:
                    class_correct[cls] += 1

    print("\n===== EVALUATION RESULTS =====")
    print(f"Overall Accuracy: {correct / total:.4f}\n")

    for cls in dataset.classes:
        acc = class_correct[cls] / max(1, class_total[cls])
        print(f"{cls:10s} : {acc:.4f}")


if __name__ == "__main__":
    main()
