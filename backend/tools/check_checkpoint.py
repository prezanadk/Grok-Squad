import os
import torch

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "weights", "cls_best.pt")

ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)

print("Checkpoint type:", type(ckpt))

# YOLOv5 typical fields
if isinstance(ckpt, dict):
    print("Keys:", list(ckpt.keys()))

    opt = ckpt.get("opt", None)
    if opt is not None:
        print("\n=== ckpt['opt'] ===")
        if isinstance(opt, dict):
            for k in ["data", "imgsz", "batch_size", "epochs", "weights", "model", "device"]:
                if k in opt:
                    print(f"{k}: {opt[k]}")
        else:
            print(opt)

    # Sometimes args are stored differently
    print("\n=== other possible fields ===")
    for k in ["train_args", "args", "cfg", "yaml", "date"]:
        if k in ckpt:
            print(f"{k}: {ckpt[k]}")
else:
    print("This checkpoint isn't a dict. It may be a raw model object.")
