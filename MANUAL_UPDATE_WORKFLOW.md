# Manual Update Workflow for Sentinel Clinics

You chose **manual update** (no auto-update server). Here's the dead-simple workflow for shipping new versions to your paying clinics.

---

## When to ship an update

- Bug fixes (security or stability)
- Better model after fine-tuning on clinic data
- New modality (e.g., adding spine MRI in V1.1)
- Per-clinic customizations (their own logo, their fine-tuned model)

Don't ship updates more than once a month — clinics don't want to babysit installs.

---

## How to build and ship one update

### Step 1 — Bump the version
```bash
cd desktop-app
# Edit package.json — change "version": "1.0.0" → "1.0.1"
# Or whatever is appropriate (semver)
```

### Step 2 — Build the installers
```bash
# From the desktop-app directory:
npm run build:win        # → release/Sentinel Medical AI Setup 1.0.1.exe
npm run build:mac        # → release/Sentinel Medical AI-1.0.1.dmg

# Both at once (only on Mac):
npm run build:all
```

Output is in `desktop-app/release/`. The Windows .exe is what most clinics install.

### Step 3 — Verify the build
```bash
# Run the smoke test against THIS machine's install
python scripts/production_smoke_test.py
# Should show 25/25 PASS

# Sanity-check the installer file exists and is the right size
ls -la desktop-app/release/Sentinel*.exe   # ~150-300 MB expected
```

### Step 4 — Upload to a place each clinic can reach
You picked manual = no fancy infra. Pick the simplest one of these:
- **WhatsApp / Telegram** to the clinic's IT — works for files <100 MB. May require splitting.
- **Google Drive shared link** — public link, file sits there until you remove it
- **WeTransfer** — 2 GB free per send, link expires in 7 days (pro for paying clinics)
- **MEGA / pCloud** — 50 GB free, permanent links
- **USB stick brought in person** — for high-trust clinics, also lets you do remote support

Don't email the .exe directly — many email gateways strip executables.

### Step 5 — Ship + brief instructions to the clinic
Use this template (copy-paste, change the version number):

```
Привет [имя],

Обновление Sentinel v1.0.1 готово.
Ссылка: [link]

Что нового:
- [главное изменение]
- [второе изменение]

Установка занимает 2 минуты:
1. Скачайте .exe файл по ссылке выше
2. Закройте Sentinel если он открыт (правый клик в трее → Exit)
3. Запустите Setup.exe и нажмите "Установить" (выберите "поверх существующей")
4. Лицензия и данные пациентов сохранятся
5. Запустите Sentinel из меню Пуск — версия 1.0.1 в "О программе"

Если что-то сломается, отправьте мне скриншот, у меня есть резервный план отката.

Best,
Шахзод
```

---

## How clinics confirm the update worked

The desktop app shows version in `Settings → About`. The first thing they check is that it shows the new version number. Tell them this in the brief.

If the installer fails or the new version crashes, the clinic falls back to the previous version (their old install isn't deleted automatically). Tell them to keep the previous installer file around for 30 days as insurance.

---

## When you need to support 20+ clinics

Manual updates start hurting around 10-15 clinics. At that point:
- **Switch to GitHub Releases** (free, public, downloads counted) — apps can pull from there
- Or **stand up a $5/month VPS** with a static file server
- Or **buy electron-updater plan** — bundled in your build, automatic update prompts

Don't pre-build this. Ship 5 paying clinics first, THEN automate.

---

## Per-clinic customization (the real win of manual updates)

Manual updates let you ship a **per-clinic build**:
- Bundle their logo into the installer
- Bundle a model fine-tuned on their data
- Default the report header to their clinic's name
- Pre-pack their license.dat in the installer

This is genuinely valuable for sales — each clinic feels they have their own product. Auto-update systems usually push the same build to everyone.

To build a per-clinic version:

```bash
# 1. Drop their logo at desktop-app/build/icon.png (replace placeholder)
# 2. Drop their fine-tuned model at desktop-app/extraResources/models/brain_finetuned/
# 3. Edit electron/main.js — set window title to include clinic name
# 4. Bundle their license.dat at build/license.dat (electron-builder copies extraResources)
# 5. Build:
cd desktop-app
PRODUCT_NAME="Sentinel — [Clinic Name]" npm run build:win
```

Charge for this — call it the "Custom" tier ($1500/month).

---

## Update versioning convention

Use semver:
- **1.0.x** = bug fix only, safe to push automatically
- **1.x.0** = new feature, brief the radiologist on what changed
- **x.0.0** = breaking change, schedule the install with the clinic

For your first 6 months you'll probably only ship 1.0.x and 1.1.0 type bumps. Reserve 2.0.0 for when you switch to subscription billing or rebuild the UI.

---

*Document by SIA Medical AI · Sentinel · May 2026*
