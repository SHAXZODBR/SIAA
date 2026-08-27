# 🌍 Cloud vs Local — Choose Your Path

## TL;DR

| For... | Use This | Why |
|--------|----------|-----|
| **Development on your Mac** | Google AI Studio (cloud) | Free, no laptop GPU usage |
| **Testing demos** | Google AI Studio (cloud) | Fast, no setup |
| **Clinic deployment** | Local Ollama | Patient data privacy law |
| **Best quality** | Local Ollama with gemma3:27b | Full control |

---

## 🎯 The Privacy Truth

**For real clinics in Uzbekistan:**
- Patient data CANNOT leave the country (medical privacy law)
- Cloud AI = data sent to Google/etc. = ILLEGAL
- Local Ollama on clinic PC = LEGAL ✅

**For development:**
- You're testing with sample/demo data
- Cloud is fine because it's not real patients
- Switch to local for production

---

## 🚀 Option A: Google AI Studio (FREE Cloud, No Laptop GPU)

### Setup (3 minutes)

1. **Get free API key:**
   - Go to https://aistudio.google.com/apikey
   - Sign in with Google account
   - Click "Create API key"
   - Copy the key

2. **Install Google AI library:**
   ```bash
   pip install google-generativeai
   ```

3. **Set environment variable:**
   ```bash
   export GOOGLE_AI_KEY="your-api-key-here"
   ```

   To make permanent, add to `~/.zshrc`:
   ```bash
   echo 'export GOOGLE_AI_KEY="your-api-key-here"' >> ~/.zshrc
   source ~/.zshrc
   ```

4. **Restart server** — auto-detects Google AI:
   ```bash
   python run_server.py
   ```

You'll see:
```
Using Google AI Studio (cloud, free tier)
Report engine initialized (backend: google_ai)
```

### Free Tier Limits

| Model | Free Tier |
|-------|-----------|
| Gemini 1.5 Flash | **15 requests/min · 1M tokens/day** |
| Gemini 1.5 Pro | 2 requests/min · 50 requests/day |
| Gemma 2 27B | Available on Vertex AI (paid) |

For Sentinel: 15 requests/min is plenty (a clinic does ~50 scans/day).

### Why Cloud Is Faster Right Now
- No 3GB download
- No GPU needed
- No setup
- Just API call

---

## 🏥 Option B: Local Ollama (Production for Clinics)

### Setup (15 minutes)

```bash
# Install (only once)
brew install ollama

# Start service
ollama serve  # leave running

# Pull model (3GB)
ollama pull gemma3:4b      # 4B fits on 4GB GPU
ollama pull gemma3:12b     # 12B better quality (8GB GPU)
ollama pull gemma3:27b     # 27B best quality (24GB RAM)
```

### Hardware Requirements

| Model | Min RAM | Min GPU | Speed | Quality |
|-------|---------|---------|-------|---------|
| gemma3:1b | 4 GB | 2 GB | Fast | Basic |
| **gemma3:4b** ⭐ | 8 GB | 4 GB | Good | **Recommended** |
| gemma3:12b | 16 GB | 8 GB | Slower | Better |
| gemma3:27b | 32 GB | 24 GB | Slow | Best |

**Your Mac M-series (M1-M4):** Can run gemma3:12b natively (Apple uses unified memory).

**Clinic GTX 1650 (4GB):** Use gemma3:4b.

---

## 🔀 Option C: Hybrid (RECOMMENDED)

```
DEVELOPMENT phase (now):
└─ Use Google AI Studio (free, no laptop usage)
    └─ Fast iteration, easy demos

PRODUCTION phase (clinic):
└─ Use local Ollama
    └─ Privacy-compliant, works offline
```

The code auto-detects which is available. You don't need to change anything.

### How auto-detection works:

```python
# Server startup tries in order:
1. Ollama (if running)         → use it
2. Google AI Studio (if key)   → use it
3. Templates (fallback)        → always works
```

So you can:
- Set GOOGLE_AI_KEY → uses cloud
- Run ollama serve → uses local (overrides cloud)
- Neither → uses templates

---

## ❓ "Can I avoid downloading entirely?"

**For Gemma reports:** YES, use Google AI Studio.

**For chest X-ray model:** Mostly no — TorchXRayVision needs ~100MB local download because:
- It's a PyTorch model (not API-based)
- Some HuggingFace alternatives exist via Inference API
- But for clinic deployment, ALL models must be local

**Alternative — Use HuggingFace Inference API for chest:**

Free tier exists but slow (10-30 sec per request). Not recommended for production.

---

## ❓ "What about Gemma 4?"

As of April 2026:
- ❌ Gemma 4 — NOT released
- ✅ Gemma 3 — Latest stable (4B, 12B, 27B)
- 🔮 Possibly Gemma 4 in late 2026 / early 2027

**When Gemma 4 releases**, just run:
```bash
ollama pull gemma4:4b    # if available
```

And update CONFIG to use it. Code is backend-agnostic.

---

## ❓ "Can I run everything on a cloud GPU server?"

YES — rent a cloud GPU and host everything there:

### Cheapest options:

| Provider | GPU | Cost | Best For |
|----------|-----|------|----------|
| **Vast.ai** | RTX 3090 | $0.30/hr | Development |
| **RunPod** | RTX 4090 | $0.44/hr | Training + dev |
| **Lambda Labs** | A10 | $0.60/hr | Production |
| **DigitalOcean** | H100 | $3.50/hr | High volume |

For development: **Vast.ai $0.30/hr × 4 hrs/day = $36/month**

But for CLINICS: must be on-premise (legal). So this is dev-only.

---

## 🎬 Right NOW — What I Recommend

### Today (development):

Skip Ollama download. Use Google AI:

```bash
# 1. Get key (2 min): https://aistudio.google.com/apikey

# 2. Install:
pip install google-generativeai

# 3. Set key:
export GOOGLE_AI_KEY="paste-your-key-here"

# 4. Start server:
cd /Users/shakhzodbtr/Desktop/Siaa_ai/sentinel
source ~/venv/bin/activate
python run_server.py
```

Server auto-detects Google AI. Reports generated in cloud. **Zero laptop GPU usage.**

### Later (clinic deployment):

When deploying to a real clinic, install Ollama locally on the clinic PC. Code already supports both — just runs whichever is available.

---

## 🆘 Concerns Addressed

| Concern | Solution |
|---------|----------|
| "I don't want to use my laptop GPU" | ✅ Use Google AI (cloud) |
| "Patient data privacy" | ✅ Local Ollama for production |
| "Want best quality" | ✅ gemma3:27b (locally) or gemini-pro (cloud) |
| "Don't want to download 3GB" | ✅ Google AI — no download |
| "Want offline capability" | ✅ Local Ollama (works without internet) |
| "Free?" | ✅ Both Google AI free tier AND Ollama are free |
| "Fastest setup?" | ✅ Google AI (3 min vs 15 min for Ollama) |

---

## 🎯 Decision Matrix

```
┌─────────────────────────────────────────────────┐
│  Are you developing/testing?                    │
│       │                                         │
│       ├─ YES → Use Google AI Studio (cloud)    │
│       │       FREE, fast, no laptop usage      │
│       │                                         │
│       └─ NO (real clinic) → Use Ollama (local) │
│                            REQUIRED for privacy │
└─────────────────────────────────────────────────┘
```

For YOU right now: **Use Google AI Studio**. Saves 30+ minutes of setup, no laptop GPU usage, everything works.

When you deploy to first clinic in 2-3 weeks: install Ollama on their PC.
