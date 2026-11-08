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
| 🧠 **Brain triage panel** | A locally validated normal-vs-abnormal triage flag — "which scans need a closer look"; unvalidated detectors (public tumor / hemorrhage classifiers) are off by default |
| 📝 **Auto-drafted reports** | Structured radiology reports in **RU / UZ / EN**; the doctor edits and signs |
| 🔒 **Fully on-premise & offline** | Patient data never leaves the clinic — built for medical trust and local law |
| 🖼️ **Robust DICOM ingestion** | Multi-frame, compressed, multi-vendor (Philips · GE · Siemens) |
| 🩺 **Radiologist-in-the-loop** | Decision support — the physician confirms every read, never autonomous |
| 🔑 **Machine-bound licensing** | RSA-signed, hardware-locked deployment; the validated model is delivered per clinic, encrypted for that one PC |

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

- ✅ **Working product (v1.1.0)** — desktop app (RU / UZ / EN interface, first-run clinic setup), inference server, whole-study DICOM ingest (thousands of files per study), multilingual reporting, PDF export, Windows installer (self-contained frozen backend + NSIS) and macOS DMG built by CI
- ✅ **Brain module** — a **locally trained normal-vs-abnormal triage model**, validated on **178 held-out Tashkent patients** (study-level sensitivity 0.90 / specificity 0.47). Unvalidated (*pending*) detectors such as the public tumor classifier are **off by default** in hospital builds and can be switched on only by an administrator (audit-logged); a head CT without a validated detector is returned as "not analysed by AI". The product **never certifies a scan as normal**.
- ✅ **Hospital-grade plumbing** — authentication on by default, server-side report signing with a tamper-evident audit log, fully offline operation, model weights encrypted at rest automatically on first start, PACS auto-ingest (Orthanc) and signed reports filed back to PACS as DICOM Encapsulated PDF, one-click PHI-free diagnostics export for support
- ✅ **Per-clinic delivery** — the clinic installer ships without the proprietary model; the vendor sends a licence plus a model bundle encrypted for that clinic's PC (`packaging/provision_clinic.py`, `docs/vendor_provisioning_en.md`), imported from *Preferences → AI models*
- ✅ **Quality gates** — CI runs the test suite (`pytest -m "not slow"`) and the documentation drift guard before any installer is built; installer artifacts expire after one day and tag builds stay draft releases; a checksummed release gate a retrained model must pass before it ships; a one-command resumable release pipeline for new hospital batches (`scripts/release_pipeline.py`, runbook in `docs/release_runbook_en.md` / `_ru.md`); hospital documentation in RU / UZ / EN (`docs/`)
- 🔧 **In progress** — code-signing certificate for the Windows installer, clinic pilot with radiologist-adjudicated evaluation
- 🗺️ **Next** — more local *normal* studies to raise specificity, **spine** (highest-volume MRI referral), then broader multi-organ coverage

> SIAA is a clinical **decision-support** tool, not an autonomous diagnostic device.
> A qualified radiologist reads and signs every study.

---

## About

**SIAA Medical AI** — building the AI radiology layer for Central Asia.
Tashkent, Uzbekistan.

Hospital documentation (user manual, admin install guide, intended use, pilot protocol) lives in [`docs/`](docs/README.md).

### → Learn more at **[www.siaa.uz](https://www.siaa.uz)**

---

<div align="center">
<sub>© SIAA Medical AI · <a href="https://www.siaa.uz">www.siaa.uz</a> · All rights reserved.</sub>
</div>
