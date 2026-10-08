"""
models.py
---------
Pydantic schemas for API request/response validation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    word: str
    start: float = Field(..., description="Start time in seconds")
    end: float   = Field(..., description="End time in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0)


class TranscriptionSegment(BaseModel):
    id: int
    text: str
    start: float = Field(..., description="Start time in seconds")
    end: float   = Field(..., description="End time in seconds")
    confidence: float
    words: List[WordTimestamp] = []


class AudioMeta(BaseModel):
    extension: str
    size_mb: float
    duration_s: float
    format: str
    sample_rate: Any
    channels: Any
    bit_rate: Any


class TranscriptionResponse(BaseModel):
    file: str
    duration_s: float
    language: str
    text: str
    segments: List[TranscriptionSegment]
    processing_time_s: float
    model: str
    chunks_processed: int
    audio_meta: AudioMeta


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    whisper_model: str
    supported_formats: List[str]
