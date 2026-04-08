from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path as _Path

from fastapi import HTTPException
from fastapi.responses import HTMLResponse

try:
    from openenv.core.env_server import create_fastapi_app
except ImportError:  # pragma: no cover
    create_fastapi_app = import_module("openenv_core.env_server").create_fastapi_app

try:
    from irce.environment import FoodCrisisEnv
    from irce.models import FoodCrisisAction, FoodCrisisObservation
except ImportError:  # pragma: no cover
    from environment import FoodCrisisEnv
    from models import FoodCrisisAction, FoodCrisisObservation

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def create_environment() -> FoodCrisisEnv:
    return FoodCrisisEnv()


app = create_fastapi_app(create_environment, FoodCrisisAction, FoodCrisisObservation)

# ─────────────────────────────────────────────────────────────────────────────
# Web UI — composed from static files at startup
# ─────────────────────────────────────────────────────────────────────────────

_STATIC_DIR = _Path(__file__).resolve().parent / "static"


def _load_html() -> str:
    """Compose the full HTML page from static CSS, HTML partials, and JS."""
    css = (_STATIC_DIR / "style.css").read_text(encoding="utf-8")
    landing = (_STATIC_DIR / "landing.html").read_text(encoding="utf-8")
    simulate = (_STATIC_DIR / "simulate.html").read_text(encoding="utf-8")
    js = (_STATIC_DIR / "app.js").read_text(encoding="utf-8")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
<meta http-equiv="Pragma" content="no-cache" />
<meta http-equiv="Expires" content="0" />
<title>FoodCrisisEnv — Food Safety Outbreak Benchmark</title>
<meta name="description" content="An OpenEnv benchmark where an AI agent acts as a food safety investigator tracing contaminated food through a supply chain before it reaches consumers.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
{css}
</style>
</head>
<body>
{landing}
{simulate}
<script>
{js}
</script>
</body>
</html>"""


_HTML = _load_html()


# ─────────────────────────────────────────────────────────────────────────────
# LLM Endpoint — Call Groq model for decisions
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/llm/test")
async def test_llm():
    """
    Test the LLM connection and list available models.
    """
    api_key = os.getenv("HF_TOKEN", "").strip()
    api_base = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1").strip()
    model = os.getenv("MODEL_NAME", "llama-3.1-8b-instant").strip()

    if not api_key:
        return {"status": "error", "message": "HF_TOKEN not set in Space Secrets"}

    try:
        client = OpenAI(api_key=api_key, base_url=api_base)

        # Try to list models (if supported)
        try:
            models = client.models.list()
            available_models = [m.id for m in models.data]
        except Exception:
            available_models = ["(unable to fetch model list)"]

        return {
            "status": "configured",
            "api_base": api_base,
            "model_requested": model,
            "available_models": available_models,
            "message": f"To use a different model, set MODEL_NAME in Space Secrets",
        }
    except Exception as e:
        return {
            "status": "error",
            "api_base": api_base,
            "model_requested": model,
            "error": str(e),
            "message": "Make sure HF_TOKEN is a valid Groq API key",
        }


@app.post("/llm/decide")
async def llm_decide(prompt: dict):
    """
    Call the LLM with a prompt and return a decision.
    Expects: {"prompt": "..."}
    Reads credentials from environment variables (HF_TOKEN, API_BASE_URL, MODEL_NAME)
    """
    import logging

    logger = logging.getLogger(__name__)

    if not OpenAI:
        raise HTTPException(status_code=500, detail="OpenAI client not installed")

    try:
        api_key = os.getenv("HF_TOKEN", "").strip()
        api_base = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1").strip()
        model = os.getenv("MODEL_NAME", "llama-3.1-8b-instant").strip()

        if not api_key:
            raise HTTPException(status_code=400, detail="HF_TOKEN not configured in Space secrets")

        logger.info(f"LLM Request: model={model}, api_base={api_base}")

        client = OpenAI(api_key=api_key, base_url=api_base)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a food safety incident responder. Respond with exactly one action from: INSPECT, QUARANTINE, LIFT, RECALL, TRACE, WAIT. Include a node or batch name if needed.",
                },
                {"role": "user", "content": prompt.get("prompt", "")},
            ],
            temperature=0.3,
            max_tokens=100,
        )
        decision = response.choices[0].message.content.strip()
        logger.info(f"LLM Response: {decision}")

        return {"action": decision}
    except Exception as e:
        import traceback

        error_msg = f"LLM error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui_root() -> HTMLResponse:
    # Ensure it reloads files on every request to avoid caching issues during dev
    html_content = _load_html() if os.getenv("IRCE_RELOAD", "false").lower() in {"1", "true", "yes"} else _HTML
    return HTMLResponse(content=html_content, status_code=200)