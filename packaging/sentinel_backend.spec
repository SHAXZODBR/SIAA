# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec — freezes the Sentinel inference backend (run_server.py) into a
# self-contained onedir bundle: packaging/dist/sentinel-backend/sentinel-backend
# Models are NOT bundled here: the installer ships models/ next to the app and the
# backend finds them via SENTINEL_MODELS_DIR (see src/utils/paths.py).
import os, sys
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

ROOT = os.path.abspath(os.path.join(os.path.dirname(SPEC), '..'))
datas, binaries, hiddenimports = [], [], []
for pkg in ('transformers', 'safetensors', 'huggingface_hub', 'tokenizers', 'pydicom', 'pylibjpeg',
            'pylibjpeg_libjpeg', 'pylibjpeg_openjpeg', 'monai', 'loguru', 'fastapi', 'starlette',
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
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='sentinel-backend', debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='sentinel-backend')
