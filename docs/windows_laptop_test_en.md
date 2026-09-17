# Windows laptop — offline test checklist (founder's copy)

Goal: build the installer on **your own Windows laptop**, install it, **switch the Wi-Fi off**, and prove
that the whole product works without internet: login → setup wizard → real brain MRI → 2 findings →
RU report → sign → PDF → language switch → restart with the study restored.
The Russian version is `windows_laptop_test_ru.md`. Print this page; tick the boxes as you go.

## 0. What you need

| | Requirement | Why |
|---|---|---|
| ☐ | Windows 10 or 11, **64-bit (x64)**, not ARM | PyInstaller + CPU PyTorch wheels are x64 only |
| ☐ | **16 GB RAM** | freezing the backend and loading the ViT models |
| ☐ | **15 GB free** on the drive that holds `C:\SIAA` | venv 3 GB + PyInstaller 4 GB + node_modules 1 GB + installer 4 GB |
| ☐ | **Python 3.11 (64-bit)** — installed once, with internet | `winget install --id Python.Python.3.11 -e --source winget` or python.org → *Windows installer (64-bit)*, tick *Add python.exe to PATH* |
| ☐ | **Node.js 20 LTS** (20.19+; 22.12+ also works, **24 does not**) | `winget install --id OpenJS.NodeJS.LTS -e --source winget` or nodejs.org → `node-v20.x.x-x64.msi` |
| ☐ | Internet **during the build only** (pip, npm, electron-builder tool downloads) | the models are already inside the kit — nothing model-related is downloaded |
| ☐ | The kit folder `SIAA_Windows_Kit` (≈ 2 GB: `sentinel\`, `demo_study\`, `models\`, `README_FIRST.txt`) | built on the Mac by `packaging/make_windows_kit.sh` |

The build script checks all of this itself and prints the exact install command when something is missing.

## 1. Build (internet ON, ~45–90 min)

1. ☐ Copy the whole kit to the laptop as **`C:\SIAA`** → the file `C:\SIAA\sentinel\packaging\build_windows.ps1` must exist.
2. ☐ Open PowerShell in `C:\SIAA\sentinel` (Explorer → Shift + right-click on empty space → *Open PowerShell window here*).
3. ☐ Run
   ```powershell
   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
   ```
   Steps 0–7 print on screen and go to `packaging\build_windows.log`. The last step (installer compression) looks frozen for 10–20 min — it is not.
4. ☐ It ends with `BUILD OK` and the installer path + size: `C:\SIAA\sentinel\desktop-app\release\Sentinel Medical AI Setup 1.0.0.exe` (expect ≈ 1.5–2 GB).
   Re-running the script is safe: finished steps are skipped.
5. ☐ If it fails: it prints the step name, the last 40 log lines and **`>>> Send this file to the developer: C:\SIAA\sentinel\packaging\build_windows.log`** — send that file. Two known hiccups: *Cannot create symbolic link* during electron-builder → Windows Settings → *For developers* → **Developer Mode ON**, re-run; *node-gyp / MSBuild* errors → you have Node 24, install Node 20 LTS.

## 2. Install (internet still ON is fine)

6. ☐ Double-click the `.exe`. SmartScreen says *Windows protected your PC* (the build is not code-signed) → **More info** → **Run anyway**.
7. ☐ The installer shows a bilingual RU/EN disclaimer → OK; keep the default folder; finish. A desktop shortcut *Sentinel Medical AI* appears.

## 3. Go offline

8. ☐ **Turn Wi-Fi OFF** (and unplug Ethernet). Confirm the network icon shows no connection. Everything below must work in this state.

## 4. First start, login, password change

9. ☐ Start *Sentinel Medical AI*. Wait up to 2–3 min the first time (Defender scans the backend once). The login screen must say the AI server is reachable (no red *AI server is not running* banner).
10. ☐ Find the **one-time admin password**. `SENTINEL_ADMIN_PASSWORD` is not set by the app, so the server generated one and printed it once. Open in Notepad:
    `%APPDATA%\sentinel-medical-ai\logs\backend.log`  (= `C:\Users\<you>\AppData\Roaming\sentinel-medical-ai\logs\backend.log` — paste the `%APPDATA%\…` path into Explorer's address bar)
    and search for `FIRST-RUN ADMIN ACCOUNT CREATED`. The line `password: …` is it. The same banner is also in `%APPDATA%\sentinel-medical-ai\logs\sentinel_YYYY-MM-DD.log`.
    Note: the installer also creates an empty `%APPDATA%\Sentinel Medical AI\` folder — ignore it; the real data folder is the lowercase, hyphenated `sentinel-medical-ai`.
11. ☐ Sign in as **`admin`** with that password.
12. ☐ The **Change password** screen appears immediately (temporary password must be replaced). Set a new one (≥ 8 characters — the app rejects shorter ones) → you land in the setup wizard.

## 5. Setup wizard (5 steps)

13. ☐ **Language** → choose Russian (default) → Next.
14. ☐ **Clinic** → type any clinic name → Next.
15. ☐ **AI server** → click **Test connection**. The *Server status* card must show **Connected**, and in *Loaded models*: **`brain_triage` — Loaded** (the required one). `brain_tumor_class` and `chest` are normally *Loaded* as well (both are warmed at start-up). → Next.
16. ☐ **License** → **Skip** (demo mode is enough for this test).
17. ☐ **Support** → Finish setup. The empty worklist appears; the top-right status must NOT say *degraded*.

## 6. The real study

18. ☐ In the worklist click **Folder** and pick **`C:\SIAA\demo_study`** (the whole folder; 23 DICOM files, series sub-folders inside). Upload + analysis take 20–90 s on CPU.
19. ☐ The study appears in the worklist; the centre viewer shows a **real brain MRI slice** (not a placeholder pattern).
20. ☐ **Findings** tab: expect **2 findings**:
    - **Abnormal (triage)** — status **Validated** (the Uzbek-trained triage model; this study's radiologist report says tumor, so it should be flagged),
    - a tumor class, e.g. **Glioma** — status **Pending validation** (public tumor classifier, not validated locally).
    The overall line reads *AI flagged a finding — requires review* and the disclaimer under it is in Russian.
21. ☐ **Report** tab: the RU report text is there (template sections: показание / методика / описание / заключение).
22. ☐ Click **Sign report** → confirm. The line *Signed by: Administrator · <date/time>* appears and the text becomes read-only.
23. ☐ Click **Export signed PDF** → choose a folder → the PDF is saved and Explorer opens on it. **Open the PDF**: Cyrillic must render (no boxes), it contains the clinic name, the finding, the signature line and the model identity `0d559766ce58`.

## 7. Languages

24. ☐ Top-right language switcher → **UZ**: the whole UI switches to Uzbek (Latin); the Report tab offers an Uzbek report.
25. ☐ Switch to **EN**: UI in English. Switch back to **RU**.

## 8. Restart

26. ☐ Close the app completely (window ✕; on Windows the backend stops too). Reopen it, sign in.
27. ☐ The study is **still in the worklist** and, once selected, shows **the same brain image** and the same findings/report (restored from the local database + `%APPDATA%\sentinel-medical-ai\previews`).
28. ☐ Still offline the whole time? Then the test is passed. Turn Wi-Fi back on.

## 9. If you see X → send me Y

| You see | Send me |
|---|---|
| `build_windows.ps1` stops with `BUILD FAILED at step …` | `C:\SIAA\sentinel\packaging\build_windows.log` |
| SmartScreen has no *Run anyway* (only *Don't run*) | screenshot; right-click the .exe → Properties → tick *Unblock* → Apply and retry |
| Login screen with red *AI server is not running* banner for > 5 min | `%APPDATA%\sentinel-medical-ai\logs\backend.log` + the newest `sentinel_*.log` from the same folder |
| A dialog *AI server restarting / stopped* | the same two logs + a screenshot of the dialog |
| No `FIRST-RUN ADMIN ACCOUNT CREATED` banner in the logs | both logs; also tell me whether `%APPDATA%\sentinel-medical-ai\sentinel.db` exists (an old database from a previous test already has an admin — delete the whole `sentinel-medical-ai` folder and restart) |
| Test connection → *degraded* / `brain_triage — Not loaded` | backend.log + a screenshot of the *Server status* card; the reason text next to the model |
| Folder upload → *No DICOM (.dcm) files in the selection* or an error toast | screenshot of the toast + which folder you picked |
| Study analysed but the viewer shows a grey placeholder / no image | screenshot + backend.log |
| Fewer or more than 2 findings, or triage not flagged | screenshot of the Findings tab (with statuses) + backend.log |
| PDF shows □□□ instead of Cyrillic, or *Could not export the PDF* | the PDF itself + screenshot |
| After restart the study is gone or has no image | backend.log + `dir %APPDATA%\sentinel-medical-ai\previews` output |
| Windows Defender / antivirus quarantines `sentinel-backend.exe` or the installer (PyInstaller false positive) | screenshot of the alert; then Windows Security → Virus & threat protection → Exclusions → add `C:\SIAA` and the install folder, re-run |
| Anything else | screenshot + the exact error text + backend.log |

Logs never contain patient names or IDs (redacted at source) — safe to send.
