"""
main.py
-------
FastAPI application – Transcription Pipeline API

Endpoints
---------
GET  /              → Web UI (index.html)
GET  /health        → Service health + model status
POST /transcribe    → Upload audio file, get transcription with timestamps
GET  /formats       → List supported audio formats

Design decisions
----------------
* UploadFile is streamed to a temp file to support large uploads without
  loading the entire file into memory.
* Transcription is synchronous here for simplicity; in production you would
  push jobs to a Celery/RQ queue and poll a /jobs/{id} endpoint.
* CORS is open (*) for demo purposes; lock it down in production.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.audio_processor import SUPPORTED_EXTENSIONS
from app.models import ErrorResponse, HealthResponse, TranscriptionResponse
from app.transcriber import get_model, transcribe_audio_file

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Audio Transcription Pipeline",
    description=(
        "Converts audio files (WAV, MP3, M4A, OGG, FLAC, …) into text with "
        "per-segment and per-word timestamps using OpenAI Whisper."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (web UI)
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Pre-load the Whisper model so the first request isn't slow."""
    logger.info("Pre-loading Whisper model on startup …")
    try:
        get_model()
        logger.info("Whisper model ready.")
    except Exception as exc:
        logger.error("Failed to load Whisper model: %s", exc)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the web UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "Audio Transcription Pipeline API", "docs": "/docs"})


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns service health and model status."""
    from app.transcriber import _whisper_model
    return HealthResponse(
        status="ok",
        model_loaded=_whisper_model is not None,
        whisper_model=os.getenv("WHISPER_MODEL", "base"),
        supported_formats=sorted(SUPPORTED_EXTENSIONS),
    )


@app.get("/formats", tags=["System"])
async def list_formats():
    """List all supported audio formats."""
    return {
        "supported_formats": sorted(SUPPORTED_EXTENSIONS),
        "note": (
            "Any format supported by ffmpeg can be converted transparently. "
            "The file is normalised to 16 kHz mono WAV before transcription."
        ),
    }


@app.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Transcription"],
    summary="Transcribe an audio file",
    description=(
        "Upload an audio file in any supported format. "
        "Returns the full transcript plus per-segment timestamps (and optional word-level timestamps). "
        "Long files are automatically split into overlapping chunks and merged."
    ),
)
async def transcribe(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Form(
        default="",
        description="ISO 639-1 language code (e.g. 'en', 'fr'). Leave empty to auto-detect.",
    ),
    task: str = Form(
        default="transcribe",
        description="'transcribe' (keep original language) or 'translate' (→ English).",
    ),
):
    # ── 1. Validate extension ─────────────────────────────────────────────────
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # ── 2. Stream upload to a temporary file ──────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="transcription_upload_") as tmpdir:
        tmp_path = os.path.join(tmpdir, filename)
        logger.info("Saving upload to %s", tmp_path)

        async with aiofiles.open(tmp_path, "wb") as out_f:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                await out_f.write(chunk)

        # ── 3. Transcribe ─────────────────────────────────────────────────────
        try:
            result = transcribe_audio_file(
                file_path=tmp_path,
                language=language.strip() if language.strip() else None,
                task=task,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected transcription error")
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {exc}",
            ) from exc

    return result
