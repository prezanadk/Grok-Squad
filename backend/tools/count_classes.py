from pathlib import Path

classes = ["akiec","bcc","bkl","df","mel","nv","vasc"]
root = Path("../../data/skin/train")
exts = {".jpg",".jpeg",".png",".bmp",".webp"}

print("Counting images in:", root.resolve())
for c in classes:
    d = root / c
    n = sum(1 for p in d.rglob("*") if p.suffix.lower() in exts) if d.exists() else 0
    print(f"{c:5s}: {n}")
