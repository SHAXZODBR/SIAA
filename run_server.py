"""Start the Sentinel inference server.

Usage:
    python run_server.py
    python run_server.py --port 8000 --host 127.0.0.1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if '--print-fingerprint' in sys.argv:
    # Used by the desktop app's setup wizard / license activation (works in the
    # PyInstaller bundle too, where there is no `python -m src.utils.license`).
    from src.utils.license import get_machine_id
    print(get_machine_id())
    sys.exit(0)

from src.inference.server import start_server


def main():
    parser = argparse.ArgumentParser(description="Sentinel — Inference Server")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    args = parser.parse_args()

    print("=" * 60)
    print("  Sentinel Medical AI — Inference Server")
    print(f"  Running on http://{args.host}:{args.port}")
    print(f"  API docs: http://{args.host}:{args.port}/docs")
    print("=" * 60)

    import uvicorn
    uvicorn.run(
        "src.inference.server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
