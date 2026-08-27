# =============================================================================
#  SENTINEL — RSNA HEMORRHAGE TRAINING on KAGGLE (paste into a Kaggle notebook)
# =============================================================================
#  Setup in Kaggle:
#    1. Accept rules: kaggle.com/c/rsna-intracranial-hemorrhage-detection/rules
#    2. New Notebook → it auto-attaches the data at
#       /kaggle/input/rsna-intracranial-hemorrhage-detection/
#    3. Settings → Accelerator → GPU T4 x2
#    4. Paste this whole file into a cell → Run All
#
#  Trains a binary hemorrhage / no-hemorrhage ViT with class-balancing, on a
#  manageable subset, and prints precision / recall / F1. Bump LIMIT for more.
#  Download the saved model dir afterward → drop into sentinel/models/head_ct_finetuned/
# =============================================================================
import os, csv, random
from collections import defaultdict
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pydicom
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ---- config ----
ROOT   = "/kaggle/input/rsna-intracranial-hemorrhage-detection"
CSV    = f"{ROOT}/stage_2_train.csv"
IMGDIR = f"{ROOT}/stage_2_train"
LIMIT  = 40000          # images to use (RSNA has ~750k; 40k is plenty to start)
EPOCHS = 5
BASE   = "google/vit-base-patch16-224"
OUT    = "/kaggle/working/head_ct_finetuned"
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(0)

# ---- labels: image_id -> any-hemorrhage 0/1 ----
lab = defaultdict(dict)
for r in csv.DictReader(open(CSV)):
    img_id, sub = r["ID"].rsplit("_", 1)
    lab[img_id][sub] = int(r["Label"])
ids = [i for i in lab if "any" in lab[i]]
random.shuffle(ids); ids = ids[:LIMIT]
print(f"using {len(ids)} images; positives = {sum(lab[i]['any'] for i in ids)}")

def window(ds, wl=40, ww=80):
    hu = ds.pixel_array.astype(np.float32) * float(getattr(ds,"RescaleSlope",1) or 1) + float(getattr(ds,"RescaleIntercept",0) or 0)
    lo, hi = wl-ww/2, wl+ww/2
    return (np.clip((hu-lo)/(hi-lo),0,1)*255).astype(np.uint8)

proc = AutoImageProcessor.from_pretrained(BASE)
CLASSES = ["hemorrhage", "no_hemorrhage"]

class RSNA(Dataset):
    def __init__(self, ids): self.ids = ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, k):
        i = self.ids[k]
        try:
            ds = pydicom.dcmread(f"{IMGDIR}/{i}.dcm", force=True)
            img = Image.fromarray(window(ds)).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224,224))
        px = proc(images=img, return_tensors="pt")["pixel_values"][0]
        y = 0 if lab[i].get("any") else 1
        return px, y

random.shuffle(ids); n_val = len(ids)//5
val_ids, train_ids = ids[:n_val], ids[n_val:]
tl = DataLoader(RSNA(train_ids), batch_size=64, shuffle=True, num_workers=2)
vl = DataLoader(RSNA(val_ids), batch_size=64, num_workers=2)

model = AutoModelForImageClassification.from_pretrained(
    BASE, num_labels=2, id2label={0:"hemorrhage",1:"no_hemorrhage"},
    label2id={"hemorrhage":0,"no_hemorrhage":1}, ignore_mismatched_sizes=True).to(device)

# class-balance (positives are rare → weight them up)
pos = sum(1 for i in train_ids if lab[i].get("any")); neg = len(train_ids)-pos
w = torch.tensor([len(train_ids)/(2*max(pos,1)), len(train_ids)/(2*max(neg,1))], device=device)
crit = nn.CrossEntropyLoss(weight=w)
opt = torch.optim.AdamW(model.parameters(), lr=3e-5)

for ep in range(EPOCHS):
    model.train()
    for bx,(x,y) in enumerate(tl):
        x,y = x.to(device), y.to(device)
        loss = crit(model(pixel_values=x).logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if bx%50==0: print(f"ep{ep} step{bx}/{len(tl)} loss {loss.item():.3f}")

# ---- eval: precision / recall / F1 ----
model.eval(); conf = defaultdict(lambda: defaultdict(int))
with torch.no_grad():
    for x,y in vl:
        p = model(pixel_values=x.to(device)).logits.argmax(-1).cpu()
        for t,pr in zip(y.tolist(), p.tolist()): conf[t][pr]+=1
print("\n=== HEMORRHAGE (val) ===")
for c,name in enumerate(CLASSES):
    tp=conf[c][c]; fn=sum(conf[c][j] for j in (0,1) if j!=c); fp=sum(conf[j][c] for j in (0,1) if j!=c)
    prec=tp/(tp+fp)*100 if tp+fp else 0; rec=tp/(tp+fn)*100 if tp+fn else 0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0
    print(f"  {name:14s} precision {prec:5.1f}%  recall {rec:5.1f}%  f1 {f1:5.1f}%")

model.save_pretrained(OUT); proc.save_pretrained(OUT)
print(f"\n✓ saved to {OUT} — download it, drop into sentinel/models/head_ct_finetuned/")
