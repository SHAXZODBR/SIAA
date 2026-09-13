# Air-gapped posture must be decided before transformers / huggingface_hub are
# imported anywhere under src/ (they read HF_HUB_OFFLINE at import time).
from src.utils import offline as _offline  # noqa: F401
