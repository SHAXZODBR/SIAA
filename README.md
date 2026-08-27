<div align="center">

# SIAA — Sentinel Medical AI

### AI-assisted radiology for the clinics of Central Asia

On-premise medical-imaging AI that reads brain **MRI / CT**, flags findings, and drafts
structured reports in **Russian · Uzbek · English** — with a radiologist confirming every result.

**🌐 [www.siaa.uz](https://www.siaa.uz)**

![Made in Uzbekistan](https://img.shields.io/badge/Made%20in-Uzbekistan-0099b5)
![On-Premise](https://img.shields.io/badge/Deployment-On--Premise%20%2F%20Offline-1C8A9B)
![Python](https://img.shields.io/badge/Backend-Python%20%2F%20FastAPI%20%2F%20PyTorch-2DD4BF)
![Desktop](https://img.shields.io/badge/App-Electron%20%2B%20React-16395F)
![Status](https://img.shields.io/badge/Status-Active%20Development-E8A33D)

</div>

---

## Why SIAA

Radiology across Central Asia faces two hard problems at once: **too few specialists** for a
fast-growing volume of scans, and **foreign AI that doesn't work here** — models trained on
Western data misread local scanners and don't speak the local languages.

**SIAA is built for the region.** It runs fully offline inside the clinic, reports in Russian,
Uzbek and English, and — critically — **learns from local patient data**, which is the one thing
no off-the-shelf model can copy.

---

## What it does

| | |
|---|---|
| 🧠 **Brain triage panel** | Flags tumor, hemorrhage, stroke and atrophy — "which scans need a closer look" |
| 📝 **Auto-drafted reports** | Structured radiology reports in **RU / UZ / EN**; the doctor edits and signs |
| 🔒 **Fully on-premise & offline** | Patient data never leaves the clinic — built for medical trust and local law |
| 🖼️ **Robust DICOM ingestion** | Multi-frame, compressed, multi-vendor (Philips · GE · Siemens) |
| 🩺 **Radiologist-in-the-loop** | Decision support — the physician confirms every read, never autonomous |
| 🔑 **Machine-bound licensing** | RSA-signed, hardware-locked deployment |

---

## How it works

```
   DICOM from scanner  →  AI triage panel  →  RU/UZ/EN report draft  →  Radiologist signs  →  PDF
        (offline)          (flags findings)      (auto-generated)          (edits & confirms)
```

Everything runs on the clinic's own machine. Nothing is uploaded.

---

## The moat 🛡️

Any team can download a public model. **Only SIAA validates and fine-tunes on real
Central-Asian patient data** — a proprietary, locally-validated dataset that is the defensible
asset behind the product. Foreign models over-call on local scanners; SIAA is trained to the
population it actually serves.

---

## Tech stack

- **Inference** — Python · FastAPI · PyTorch · MONAI · pydicom
- **Desktop app** — Electron · React · TypeScript · Vite
- **Reporting** — local LLM (Gemma via Ollama) with RU / UZ / EN templates
- **Security** — RSA-signed machine licensing · JWT auth · audit logging

---

## Status & roadmap

- ✅ **Working product** — desktop app, inference server, multilingual reporting, DICOM pipeline
- ✅ **Brain module** — triage panel trained & validated on real local patient studies
- 🔧 **In progress** — expanding brain findings, whole-study validation, clinic pilot
- 🗺️ **Next** — **spine** (highest-volume MRI referral), then broader multi-organ coverage

> SIAA is a clinical **decision-support** tool, not an autonomous diagnostic device.
> A qualified radiologist reads and signs every study.

---

## About

**SIAA Medical AI** — building the AI radiology layer for Central Asia.
Tashkent, Uzbekistan.

### → Learn more at **[www.siaa.uz](https://www.siaa.uz)**

---

<div align="center">
<sub>© SIAA Medical AI · <a href="https://www.siaa.uz">www.siaa.uz</a> · All rights reserved.</sub>
</div>
