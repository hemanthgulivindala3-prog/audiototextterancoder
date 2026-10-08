"""
transcriber.py
--------------
Core transcription engine backed by OpenAI Whisper (open-source).

Design decisions
----------------
* Model is loaded once at startup and cached (singleton) to avoid re-loading
  the large weights on every request.
* We use Whisper's `transcribe()` with `word_timestamps=True` so we can
  expose fine-grained word-level timing in addition to segment timestamps.
* For multi-chunk long files the chunk offsets are added to every segment /
  word timestamp to give absolute times.
* The overlap deduplication is handled by preferring the first chunk's
  transcription of any overlapping region (simple heuristic; a more robust
  approach would use edit-distance alignment).
* Model size is configurable via the WHISPER_MODEL env var:
    tiny | base | small | medium | large | large-v2 | large-v3
  Default is "base" which is a good balance of speed and accuracy.
"""

import os
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import whisper

from app.audio_processor import (
    load_and_normalise,
    split_audio,
    export_chunk_to_wav,
    validate_audio_file,
    CHUNK_OVERLAP_S,
    WHISPER_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)

# ── Model singleton ───────────────────────────────────────────────────────────
_whisper_model: Optional[whisper.Whisper] = None


def get_model() -> whisper.Whisper:
    """Return the cached Whisper model, loading it on first call."""
    global _whisper_model
    if _whisper_model is None:
        model_name = os.getenv("WHISPER_MODEL", "tiny")
        logger.info("Loading Whisper model '%s' …", model_name)
        t0 = time.time()
        _whisper_model = whisper.load_model(model_name)
        logger.info("Whisper model loaded in %.1f s", time.time() - t0)
    return _whisper_model


# ── Data models (plain dicts – no Pydantic dependency here) ──────────────────

def _make_word(word: str, start: float, end: float, probability: float) -> Dict[str, Any]:
    return {
        "word": word.strip(),
        "start": round(start, 3),
        "end": round(end, 3),
        "confidence": round(probability, 4),
    }


def _make_segment(
    segment_id: int,
    text: str,
    start: float,
    end: float,
    avg_logprob: float,
    words: List[Dict],
) -> Dict[str, Any]:
    return {
        "id": segment_id,
        "text": text.strip(),
        "start": round(start, 3),
        "end": round(end, 3),
        "confidence": round(float(2 ** avg_logprob), 4),   # logprob → probability
        "words": words,
    }


# ── Core transcription helpers ────────────────────────────────────────────────

def _transcribe_wav(
    wav_path: str,
    language: Optional[str],
    task: str,
    time_offset: float,
    seg_id_offset: int,
) -> Dict[str, Any]:
    """
    Transcribe a single WAV file and return normalised segment/word data.
    All timestamps are shifted by *time_offset* seconds.
    """
    model = get_model()

    options: Dict[str, Any] = {
        "task": task,                   # "transcribe" or "translate"
        "word_timestamps": True,
        "verbose": False,
        "fp16": False,
    }
    if language:
        options["language"] = language

    result = model.transcribe(wav_path, **options)

    segments = []
    seg_id = seg_id_offset

    for raw_seg in result.get("segments", []):
        raw_words = raw_seg.get("words", [])
        words = [
            _make_word(
                w["word"],
                w["start"] + time_offset,
                w["end"]   + time_offset,
                w.get("probability", 0.0),
            )
            for w in raw_words
        ]
        seg = _make_segment(
            segment_id = seg_id,
            text       = raw_seg["text"],
            start      = raw_seg["start"] + time_offset,
            end        = raw_seg["end"]   + time_offset,
            avg_logprob= raw_seg.get("avg_logprob", -1.0),
            words      = words,
        )
        segments.append(seg)
        seg_id += 1

    return {
        "language": result.get("language", "unknown"),
        "segments": segments,
    }


def _merge_chunks(
    chunk_results: List[Dict[str, Any]],
    overlap_s: float,
) -> List[Dict[str, Any]]:
    """
    Merge transcription results from overlapping chunks.

    Strategy: when the start of a segment in chunk N+1 falls within the
    overlap region of chunk N, we skip that segment (it was already captured
    by chunk N). This is a conservative approach that avoids duplicates at the
    cost of occasionally missing a word if the boundary falls awkwardly.
    """
    if not chunk_results:
        return []

    merged: List[Dict[str, Any]] = list(chunk_results[0]["segments"])
    seg_id_counter = len(merged)

    for prev_result, curr_result in zip(chunk_results, chunk_results[1:]):
        if not prev_result["segments"]:
            # nothing to compare against; take all current segments
            for seg in curr_result["segments"]:
                seg = dict(seg, id=seg_id_counter)
                merged.append(seg)
                seg_id_counter += 1
            continue

        prev_end = prev_result["segments"][-1]["end"]
        dedup_boundary = prev_end - overlap_s

        for seg in curr_result["segments"]:
            if seg["start"] < dedup_boundary:
                continue    # duplicate from overlap region
            seg = dict(seg, id=seg_id_counter)
            merged.append(seg)
            seg_id_counter += 1

    return merged


# ── Public API ────────────────────────────────────────────────────────────────

def transcribe_audio_file(
    file_path: str,
    language: Optional[str] = None,
    task: str = "transcribe",
) -> Dict[str, Any]:
    """
    Full transcription pipeline for a single audio file.

    Parameters
    ----------
    file_path : str  – Path to the input audio file (any supported format).
    language  : str  – ISO 639-1 language code, e.g. "en". None = auto-detect.
    task      : str  – "transcribe" (default) or "translate" (→ English).

    Returns
    -------
    {
        "file": str,
        "duration_s": float,
        "language": str,
        "text": str,              # full concatenated transcript
        "segments": [...],        # list of segment dicts with timestamps
        "processing_time_s": float,
        "model": str,
        "chunks_processed": int,
    }
    """
    t_start = time.time()

    # 1. Validate
    meta = validate_audio_file(file_path)
    logger.info("Transcribing %s (%.1f s, %.1f MB)", file_path, meta["duration_s"], meta["size_mb"])

    # 2. Load & normalise to 16 kHz mono float32 numpy array
    audio, sample_rate = load_and_normalise(file_path)

    # 3. Split into chunks if needed
    chunks = split_audio(audio, sample_rate)

    chunk_results: List[Dict[str, Any]] = []
    detected_language = language or "unknown"
    seg_id_offset = 0

    # 4. Transcribe each chunk
    with tempfile.TemporaryDirectory(prefix="whisper_chunks_") as tmpdir:
        for idx, (chunk_audio, offset_s) in enumerate(chunks):
            logger.info(
                "Transcribing chunk %d/%d (offset=%.1f s) …",
                idx + 1, len(chunks), offset_s,
            )
            wav_path = export_chunk_to_wav(chunk_audio, sample_rate, tmpdir)
            chunk_res = _transcribe_wav(
                wav_path      = wav_path,
                language      = language,
                task          = task,
                time_offset   = offset_s,
                seg_id_offset = seg_id_offset,
            )
            # Use the language detected in the first chunk
            if idx == 0:
                detected_language = chunk_res["language"]

            chunk_results.append(chunk_res)
            seg_id_offset += len(chunk_res["segments"])

    # 5. Merge overlapping chunks
    merged_segments = _merge_chunks(chunk_results, float(CHUNK_OVERLAP_S))

    # 6. Build full text
    full_text = " ".join(seg["text"] for seg in merged_segments)

    return {
        "file": Path(file_path).name,
        "duration_s": meta["duration_s"],
        "language": detected_language,
        "text": full_text.strip(),
        "segments": merged_segments,
        "processing_time_s": round(time.time() - t_start, 2),
        "model": os.getenv("WHISPER_MODEL", "tiny"),
        "chunks_processed": len(chunks),
        "audio_meta": meta,
    }
