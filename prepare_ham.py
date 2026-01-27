import os, shutil
import pandas as pd

RAW_DIR = "ham10000_raw"
META = os.path.join(RAW_DIR, "HAM10000_metadata.csv")

P1 = os.path.join(RAW_DIR, "HAM10000_images_part_1")
P2 = os.path.join(RAW_DIR, "HAM10000_images_part_2")

OUT_BENIGN = "data/benign"
OUT_MALIGN = "data/malignant"

# Standard binary grouping used in many HAM10000 baselines:
# malignant: melanoma, basal cell carcinoma, actinic keratoses/intraepithelial carcinoma
malignant = {"mel", "bcc", "akiec"}
benign = {"nv", "bkl", "df", "vasc"}

os.makedirs(OUT_BENIGN, exist_ok=True)
os.makedirs(OUT_MALIGN, exist_ok=True)

df = pd.read_csv(META)

missing = 0
skipped = 0

for _, row in df.iterrows():
    img_id = row["image_id"]
    dx = row["dx"]

    src1 = os.path.join(P1, img_id + ".jpg")
    src2 = os.path.join(P2, img_id + ".jpg")
    src = src1 if os.path.exists(src1) else (src2 if os.path.exists(src2) else None)

    if src is None:
        missing += 1
        continue

    if dx in malignant:
        dst = os.path.join(OUT_MALIGN, img_id + ".jpg")
    elif dx in benign:
        dst = os.path.join(OUT_BENIGN, img_id + ".jpg")
    else:
        skipped += 1
        continue

    shutil.copyfile(src, dst)

print("✅ Done sorting HAM10000")
print("Missing images:", missing)
print("Skipped labels:", skipped)
print("Benign:", len(os.listdir(OUT_BENIGN)))
print("Malignant:", len(os.listdir(OUT_MALIGN)))
