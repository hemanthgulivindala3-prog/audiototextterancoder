"""
audio_processor.py
------------------
Handles:
  - Audio format detection and conversion (WAV, MP3, M4A, OGG, FLAC, WEBM, …)
  - Validation (file size, duration limits)
  - Chunking strategy for long audio files
  - Normalisation (mono, 16 kHz) required by Whisper

Design decisions
----------------
* Uses ffmpeg (via subprocess) for format-agnostic loading — avoids pydub's
  audioop dependency which was removed in Python 3.13+.
* numpy arrays are used for audio data manipulation (slicing, channel merging).
* soundfile handles WAV I/O for the normalised chunks.
* Chunks overlap by CHUNK_OVERLAP_S so we don't lose words at segment
  boundaries; the transcriber strips duplicate words at joins.
* All temporary files live inside a caller-supplied tmp directory and are
  cleaned up by the caller (or the OS tempfile machinery).
"""

import os
import math
import json
import logging
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}
MAX_FILE_SIZE_MB     = 500
CHUNK_DURATION_S     = 10 * 60      # 10 minutes
CHUNK_OVERLAP_S      = 3            # 3-second overlap
WHISPER_SAMPLE_RATE  = 16_000       # Hz
WHISPER_CHANNELS     = 1            # mono


def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        import os
        venv_scripts = os.path.join(os.path.dirname(os.path.dirname(__file__)), "venv", "Scripts")
        os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH. "
                "Please install ffmpeg from https://ffmpeg.org and add it to your PATH."
            )


# ── Public helpers ────────────────────────────────────────────────────────────

def validate_audio_file(file_path: str) -> dict:
    """
    Lightweight validation before loading the full audio.
    Returns a metadata dict on success; raises ValueError on failure.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    size_mb = path.stat().st_size / (1024 ** 2)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large ({size_mb:.1f} MB). Limit is {MAX_FILE_SIZE_MB} MB.")

    # Use ffprobe for metadata
    _check_ffmpeg()
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-show_format",
                file_path,
            ],
            capture_output=True, text=True, check=True,
        )
        info = json.loads(result.stdout)
        fmt     = info.get("format", {})
        streams = info.get("streams", [])
        audio   = next((s for s in streams if s.get("codec_type") == "audio"), {})
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        # ffprobe not available — fall back to basic info
        fmt, audio = {}, {}

    duration_s = float(fmt.get("duration") or audio.get("duration") or 0)
    return {
        "extension":   ext,
        "size_mb":     round(size_mb, 2),
        "duration_s":  round(duration_s, 2),
        "format":      fmt.get("format_name", "unknown"),
        "sample_rate": audio.get("sample_rate", "unknown"),
        "channels":    audio.get("channels", "unknown"),
        "bit_rate":    fmt.get("bit_rate", "unknown"),
    }


def load_and_normalise(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load any supported audio format via ffmpeg and normalise to 16 kHz mono
    float32 numpy array — the exact format Whisper expects.

    Returns (audio_array, sample_rate) where audio_array is shape (N,).
    """
    _check_ffmpeg()
    logger.info("Loading & normalising: %s", file_path)

    # Pipe raw PCM out of ffmpeg: 16-bit signed, 16 kHz, mono
    cmd = [
        "ffmpeg",
        "-i", file_path,
        "-ar", str(WHISPER_SAMPLE_RATE),
        "-ac", "1",
        "-f", "s16le",          # signed 16-bit little-endian raw PCM
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise ValueError(
            f"ffmpeg failed to decode '{file_path}': {result.stderr.decode(errors='replace')[:300]}"
        )

    # Convert raw bytes → float32 in [-1, 1]
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, WHISPER_SAMPLE_RATE


def split_audio(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration_s: int = CHUNK_DURATION_S,
    overlap_s: int = CHUNK_OVERLAP_S,
) -> List[Tuple[np.ndarray, float]]:
    """
    Split a long audio array into overlapping chunks.

    Returns a list of (chunk_array, start_offset_seconds) tuples so the
    transcriber can adjust timestamps from relative → absolute.
    Short files (≤ chunk_duration_s) are returned as a single chunk.
    """
    total_samples     = len(audio)
    chunk_samples     = chunk_duration_s * sample_rate
    overlap_samples   = overlap_s * sample_rate
    step_samples      = chunk_samples - overlap_samples

    if total_samples <= chunk_samples:
        return [(audio, 0.0)]

    n_chunks = math.ceil((total_samples - overlap_samples) / step_samples)
    logger.info(
        "Splitting %.1f-min audio into %d chunks (chunk=%d min, overlap=%d s)",
        total_samples / sample_rate / 60, n_chunks,
        chunk_duration_s // 60, overlap_s,
    )

    chunks: List[Tuple[np.ndarray, float]] = []
    for i in range(n_chunks):
        start = i * step_samples
        end   = min(start + chunk_samples, total_samples)
        chunks.append((audio[start:end], start / sample_rate))

    return chunks


def export_chunk_to_wav(chunk: np.ndarray, sample_rate: int, directory: str) -> str:
    """
    Export an audio numpy array to a temporary WAV file in *directory*.
    Returns the absolute path of the created file.
    """
    fd, path = tempfile.mkstemp(suffix=".wav", dir=directory)
    os.close(fd)
    sf.write(path, chunk, sample_rate, subtype="PCM_16")
    return path
