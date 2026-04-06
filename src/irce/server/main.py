from __future__ import annotations

import os

import uvicorn


def main() -> None:
    app_path = "irce.server.app:app"

    try:
        __import__("irce.server.app")
    except ImportError:  # pragma: no cover
        app_path = "server.app:app"

    uvicorn.run(
        app_path,
        host=os.getenv("FOODCRISIS_HOST", "0.0.0.0"),
        port=int(os.getenv("FOODCRISIS_PORT", "7860")),
        reload=os.getenv("FOODCRISIS_RELOAD", "false").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    main()
