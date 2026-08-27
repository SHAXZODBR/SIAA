# 🆓 FREE GPU Training — Complete Step-by-Step

## 🏆 Best Free Option: **Kaggle Notebooks**

Kaggle gives you:
- ✅ **Free T4 x2 GPUs (32GB total VRAM)** — better than Colab free
- ✅ **30 hours/week** of GPU time
- ✅ **NIH + RSNA datasets already available** — no upload needed!
- ✅ **No credit card required**
- ✅ Persistent outputs you can download

---

## 📋 KAGGLE STEP-BY-STEP

### Step 1: Create Kaggle Account (3 min)

1. Go to **https://www.kaggle.com**
2. Click **Register** → use Google account (fastest)
3. Verify your email

### Step 2: Verify Phone Number (REQUIRED for GPU access)

1. Go to **https://www.kaggle.com/settings**
2. Scroll to **Phone Verification**
3. Enter your phone → receive SMS code → verify
4. **This unlocks GPU access** (one-time, no charges)

### Step 3: Add Your SSH key (for downloading)

Skip this — Kaggle provides web UI download.

### Step 4: Create Notebook

1. Go to **https://www.kaggle.com/code**
2. Click **+ New Notebook**
3. In right sidebar (click the arrow if hidden):
   - **Accelerator:** Select **GPU T4 x2**
   - **Persistence:** Select **Files only**
   - **Internet:** **On** (to download pretrained weights)

### Step 5: Add Datasets (CRITICAL — FREE AUTOMATIC)

In the right sidebar, click **+ Add Data**:

1. **Search: `nih-chest-xrays`**
   - Click **NIH Chest X-rays** (by National Institutes of Health)
   - Click **Add** → 45 GB loaded instantly to `/kaggle/input/data/`

2. **Search: `rsna-pneumonia-detection`**
   - Click **RSNA Pneumonia Detection Challenge**
   - Click **Add** → 10 GB loaded to `/kaggle/input/rsna-pneumonia-detection-challenge/`

**You just loaded 55 GB of medical imaging data in 10 seconds — completely free. No upload needed.**

### Step 6: Paste Training Script

1. In the notebook, **delete the default cell**
2. Copy the ENTIRE contents of `sentinel/training/kaggle_train.py`
3. Paste into one cell
4. Click the **▶ Run** button (or press Shift+Enter)

### Step 7: Go Do Something Else (6-8 hours)

Training runs automatically. You can:
- Close the browser tab (Kaggle keeps running)
- Go to bed
- Come back later

**⚠️ Keep at least one tab open for the session.** If you close ALL Kaggle tabs, it might timeout after a while.

### Step 8: Monitor Progress (Optional)

Open your Kaggle notebook anytime at:
**https://www.kaggle.com/code/YOUR_USERNAME/YOUR_NOTEBOOK_NAME**

You'll see live output showing:
```
E  1 | loss=0.3421 vl=0.2934 | recall=0.645 prec=0.587 auc=0.821 | 1240s
E  2 | loss=0.2534 vl=0.2103 | recall=0.731 prec=0.642 auc=0.867 | 1185s
...
E 15 | loss=0.1203 vl=0.1845 | recall=0.892 prec=0.721 auc=0.924 | 1192s
```

### Step 9: Download Trained Model

When training finishes (~6-8 hours), in the notebook:

1. Click the **Output** tab at the top
2. You'll see these files:
   - `best_model.pt` ← **DOWNLOAD THIS** (~100 MB)
   - `results.json`
   - `training_curves.png`
   - `per_class_metrics.png`
3. Click **Download All** (or download individually)

### Step 10: Deploy to Your Mac

1. Move `best_model.pt` to:
   ```
   /Users/shakhzodbtr/Desktop/Siaa_ai/sentinel/models/densenet/best_model.pt
   ```

2. Restart your Python server:
   ```bash
   cd /Users/shakhzodbtr/Desktop/Siaa_ai/sentinel
   source ~/venv/bin/activate
   python3 run_server.py
   ```

3. Server output should now say:
   ```
   DenseNet121 loaded from models/densenet/best_model.pt (14 classes)
   Loaded optimized thresholds for 14 classes
   ```

4. **Upload a real DICOM** to the desktop app — you'll get REAL AI analysis!

---

## 🥈 Alternative: Google Colab

Colab is **less convenient** because you must upload data to Google Drive first.

### Colab Steps:

1. **Upload data to Google Drive** (overnight)
   - Go to **https://drive.google.com**
   - Create folder: `sentinel_data`
   - Upload `nih-chestxray14/` (42 GB)
   - Upload `rsna-pneumonia/` (10 GB)

2. **Open Colab**: **https://colab.research.google.com**

3. **Runtime → Change runtime type → GPU (T4)**

4. **First cell** (setup):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !pip install -q pydicom loguru
   import torch
   print('GPU:', torch.cuda.get_device_name(0))
   ```

5. **Second cell**: paste contents of `kaggle_train.py` but change paths:
   ```python
   CONFIG = {
       'nih_dir': '/content/drive/MyDrive/sentinel_data/nih-chestxray14',
       'rsna_dir': '/content/drive/MyDrive/sentinel_data/rsna-pneumonia',
       ...
   }
   ```

6. **Run** — downloads model to Drive

### ⚠️ Colab Limitations:
- Session disconnects after **90 min idle** OR **12 hours max**
- You need to **keep a tab open** and click "I'm still here"
- Free T4 is only **15 GB VRAM** (vs Kaggle's 32 GB T4 x2)

---

## 🥉 Google Cloud Platform ($300 Free Credit)

Only use this if you want a LONG training run (20+ hrs).

1. **Sign up**: https://cloud.google.com/free
2. Verify with credit card (for free credit)
3. **Get $300 credit** valid for 90 days
4. Create **Vertex AI Workbench** or **Compute Engine VM** with GPU
5. More complex — skip unless you have specific needs

---

## 💎 PRO TIPS for Kaggle Training

### Speed Up Training
The master script uses `image_size=320` instead of 512.
For **faster experimentation**, set:
```python
CONFIG['image_size'] = 256   # Reduces from ~8hr to ~5hr
CONFIG['max_train_samples'] = 20000  # Quick test run (~1hr)
```

### Prevent Timeout
Kaggle sessions timeout after **9 hours** by default. To prevent this:
1. Settings → **Session Length** → **12 hours** (max)
2. Setup **Kaggle Session Keep-Alive** via browser extension (optional)

### If It Fails, Resume
The script saves checkpoints automatically. To resume:
- Rerun the notebook — it finds the checkpoint and continues

### Better Results (More Epochs)
If recall is below 85% after first run:
```python
CONFIG['epochs'] = 30        # More training
CONFIG['warmup_epochs'] = 3  # Smoother startup
```

---

## 📊 Expected Results

After training with Kaggle T4 x2:

| Metric | Expected |
|--------|---------|
| Training time | 6-8 hours |
| Macro Recall | 85-90% |
| Macro AUC-ROC | 0.85-0.92 |
| Model size | ~100 MB |
| Inference speed on GTX 1650 | ~2-3 sec/image |

---

## ⚡ Quick Start TL;DR

```
1. Kaggle.com → Register
2. Settings → Verify phone
3. Code → New Notebook
4. Right sidebar → GPU T4 x2 + Add Data (NIH + RSNA)
5. Paste kaggle_train.py → Run All
6. Wait 8 hours
7. Output tab → Download best_model.pt
8. Copy to sentinel/models/densenet/best_model.pt
9. Restart Python server — REAL AI now works!
```

---

## 🆘 Troubleshooting

### "GPU not available"
→ Settings → Accelerator → GPU T4 x2 → Save

### "Dataset not found"
→ Add Data button → search "nih-chest-xrays" and "rsna-pneumonia-detection"

### "Session disconnected"
→ Kaggle keeps running. Refresh page, outputs still there.

### "Out of memory"
→ Reduce `CONFIG['batch_size'] = 16` (from 32)

### "Taking too long"
→ Reduce `CONFIG['max_train_samples'] = 30000` for faster test

### Model downloads but server can't load it
→ Make sure path is exactly:
   `sentinel/models/densenet/best_model.pt`

---

## 💰 Final Cost: $0

- Kaggle notebook: **FREE**
- NIH dataset: **FREE** (on Kaggle)
- RSNA dataset: **FREE** (on Kaggle)
- GPU compute: **FREE** (30 hrs/week)
- Your trained model: **YOURS forever**

**Total: $0. Total time: ~1 day (8 hours of waiting).**
