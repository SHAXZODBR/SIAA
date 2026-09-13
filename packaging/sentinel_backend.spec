# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec — freezes the Sentinel inference backend (run_server.py) into a
# self-contained onedir bundle: packaging/dist/sentinel-backend/sentinel-backend
# (sentinel-backend.exe on Windows — the name desktop-app/electron/main.js looks for).
# PyInstaller does not cross-compile: run packaging/build_backend.sh on macOS and
# packaging/build_backend.ps1 on Windows (.github/workflows/build-windows.yml does both).
# Models are NOT bundled here: the installer ships models/ next to the app and the
# backend finds them via SENTINEL_MODELS_DIR (see src/utils/paths.py).
# console=True is intentional on every OS: the Electron shell spawns the exe with
# piped stdio and appends stdout/stderr to <data dir>/logs/backend.log.
import os, sys
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

ROOT = os.path.abspath(os.path.join(os.path.dirname(SPEC), '..'))
datas, binaries, hiddenimports = [], [], []
# pylibjpeg plugins are discovered through entry-point metadata, which collect_all
# copies — list every decoder from requirements.txt so compressed DICOM (JPEG2000 /
# JPEG-LS / RLE) decodes inside the bundle on Windows exactly as it does from a venv.
# imageio (pulled in by torchxrayvision) reads its own dist-info at import: without its
# metadata the whole chest model fails with "No package metadata was found for imageio".
for pkg in ('transformers', 'safetensors', 'huggingface_hub', 'tokenizers', 'pydicom', 'pylibjpeg',
            'pylibjpeg_libjpeg', 'pylibjpeg_openjpeg', 'pylibjpeg_rle', 'gdcm', 'imageio',
            'monai', 'loguru', 'fastapi', 'starlette',
            'uvicorn', 'pydantic', 'pydantic_core', 'anyio', 'cryptography', 'PIL', 'numpy',
            'torch', 'torchvision', 'torchxrayvision', 'jose', 'bcrypt', 'python_multipart',
            'multipart', 'yaml', 'requests', 'httpx', 'scipy', 'skimage', 'cv2', 'sklearn', 'nibabel'):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass
hiddenimports += collect_submodules('src')
hiddenimports += ['uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto', 'uvicorn.protocols',
                  'uvicorn.protocols.http', 'uvicorn.protocols.http.auto', 'uvicorn.protocols.websockets',
                  'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan', 'uvicorn.lifespan.on']
datas += [(os.path.join(ROOT, 'configs'), 'configs')]

a = Analysis([os.path.join(ROOT, 'run_server.py')], pathex=[ROOT], binaries=binaries, datas=datas,
             hiddenimports=hiddenimports, hookspath=[], runtime_hooks=[],
             excludes=['matplotlib', 'tkinter', 'IPython', 'jupyter', 'notebook', 'pytest', 'tensorflow',
                       'jax', 'flax', 'onnxruntime', 'triton'],
             noarchive=False)
pyz = PYZ(a.pure)
# Windows only: stamp the desktop app's icon on sentinel-backend.exe (Task Manager /
# Explorer). macOS keeps PyInstaller's default — the .app icon comes from electron-builder.
WIN_ICON = os.path.join(ROOT, 'desktop-app', 'build', 'icon.ico')
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='sentinel-backend', debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=True,
          icon=WIN_ICON if (sys.platform == 'win32' and os.path.exists(WIN_ICON)) else None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='sentinel-backend')
