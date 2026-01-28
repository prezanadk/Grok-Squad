from pathlib import Path
import random
import shutil

# --------------------
# CONFIG
# --------------------
SRC_TRAIN = Path("../../data/skin/train")
SRC_VAL   = Path("../../data/skin/val")
OUT_ROOT  = Path("../../data/skin_train_balanced")

CLASSES = ["akiec","bcc","bkl","df","mel","nv","vasc"]
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}
SEED = 42

random.seed(SEED)

def list_imgs(d: Path):
    return [p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS]

def main():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    print("Creating balanced dataset...")

    # ---- copy validation unchanged ----
    for c in CLASSES:
        (OUT_ROOT / "val" / c).mkdir(parents=True, exist_ok=True)
        for p in list_imgs(SRC_VAL / c):
            shutil.copy2(p, OUT_ROOT / "val" / c / p.name)

    # ---- read train counts ----
    train_imgs = {c: list_imgs(SRC_TRAIN / c) for c in CLASSES}
    counts = {c: len(train_imgs[c]) for c in CLASSES}

    target = max(counts.values())  # = nv count
    print("Target per class:", target)
    print("Original counts:", counts)

    # ---- oversample train ----
    for c in CLASSES:
        out_dir = OUT_ROOT / "train" / c
        out_dir.mkdir(parents=True, exist_ok=True)

        imgs = train_imgs[c]
        if not imgs:
            print("Skipping empty:", c)
            continue

        # copy originals
        for p in imgs:
            shutil.copy2(p, out_dir / p.name)

        # duplicate until target reached
        i = 0
        while len(list_imgs(out_dir)) < target:
            p = random.choice(imgs)
            dst = out_dir / f"{p.stem}_dup{i}{p.suffix}"
            shutil.copy2(p, dst)
            i += 1

        print(f"{c}: {len(list_imgs(out_dir))}")

    print("\nBalanced dataset created at:")
    print(OUT_ROOT.resolve())

if __name__ == "__main__":
    main()
