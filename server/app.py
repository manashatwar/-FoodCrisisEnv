from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irce.server.app import app

__all__ = ["app", "main"]


def main() -> None:
    uvicorn.run(
        "server.app:app",
        host=os.getenv("IRCE_HOST", "0.0.0.0"),
        port=int(os.getenv("IRCE_PORT", "7860")),
        reload=os.getenv("IRCE_RELOAD", "false").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    main()
