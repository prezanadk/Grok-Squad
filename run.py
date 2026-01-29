import os
import random
import shutil

# ======================
# CONFIG
# ======================
TRAIN_DIR = "data/train"
VAL_DIR   = "data/val"

VAL_RATIO = 0.2   # 20% goes to validation
SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
random.seed(SEED)

os.makedirs(VAL_DIR, exist_ok=True)

# ======================
# SPLIT TRAIN -> VAL (MOVE files)
# ======================
for cls in os.listdir(TRAIN_DIR):
    cls_train = os.path.join(TRAIN_DIR, cls)
    if not os.path.isdir(cls_train):
        continue

    images = [
        f for f in os.listdir(cls_train)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]

    if len(images) == 0:
        print(f"[SKIP] {cls}: no images")
        continue

    random.shuffle(images)

    # keep at least 1 image in val if class is big enough
    val_count = int(len(images) * VAL_RATIO)
    if len(images) >= 5:
        val_count = max(1, val_count)
    else:
        val_count = 0

    cls_val = os.path.join(VAL_DIR, cls)
    os.makedirs(cls_val, exist_ok=True)

    moved = 0
    for f in images[:val_count]:
        shutil.move(
            os.path.join(cls_train, f),
            os.path.join(cls_val, f)
        )
        moved += 1

    print(f"{cls:6s} | moved to val: {moved:4d} | left in train: {len(images)-moved:4d}")

print("\n[DONE] Created data/val from data/train")
