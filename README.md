# 🎙️ AudioScript — Audio Transcription Pipeline

A production-quality audio-to-text transcription pipeline powered by **OpenAI Whisper**.  
Accepts any audio format, handles long files automatically, and returns timestamped transcripts.

---

## Features

| Feature | Detail |
|---|---|
| **Multi-format support** | WAV, MP3, M4A, OGG, FLAC, WebM, AAC, OPUS (via ffmpeg) |
| **Timestamped output** | Segment-level *and* word-level timestamps |
| **Long file handling** | Automatic chunking with 3-second overlap + deduplication |
| **Language auto-detect** | Whisper detects language; supports 99+ languages |
| **Translation** | Optional English translation via Whisper's translate task |
| **REST API** | FastAPI with OpenAPI docs at `/docs` |
| **Web UI** | Drag-and-drop upload, segment search, JSON/TXT download |
| **CLI** | Direct transcription without starting the web server |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       FastAPI Server                         │
│                                                              │
│   POST /transcribe                                           │
│         │                                                    │
│         ▼                                                    │
│   audio_processor.py                                         │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ 1. validate_audio_file()  — extension + size check  │   │
│   │ 2. load_and_normalise()   — ffmpeg → 16kHz mono WAV │   │
│   │ 3. split_audio()          — 10-min chunks, 3s overlap│   │
│   └─────────────────────────────────────────────────────┘   │
│         │                                                    │
│         ▼  (for each chunk)                                  │
│   transcriber.py                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ 4. export_chunk_to_wav()  — temp WAV file           │   │
│   │ 5. whisper.transcribe()   — word_timestamps=True    │   │
│   │ 6. _merge_chunks()        — offset correction       │   │
│   │                             + overlap deduplication  │   │
│   └─────────────────────────────────────────────────────┘   │
│         │                                                    │
│         ▼                                                    │
│   JSON Response: { text, segments, words, metadata }        │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.9+
- [ffmpeg](https://ffmpeg.org/download.html) installed and on your PATH

### 1. Create a virtual environment and install dependencies

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

> **Note:** `torch` will be installed automatically. For GPU support, install the appropriate CUDA version of PyTorch first.

### 2. Start the server

```bash
python run.py
```

Or with uvicorn directly:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

### 3. Use the CLI (no server needed)

```bash
# Basic transcription
python cli.py audio.mp3

# With options
python cli.py audio.wav --model medium --language en --output result.json

# Translation to English
python cli.py audio.fr.mp3 --task translate

# Plain text only
python cli.py audio.mp3 --text-only
```

---

## API Reference

### `POST /transcribe`

| Parameter | Type | Description |
|---|---|---|
| `file` | file | Audio file (multipart/form-data) |
| `language` | string | ISO 639-1 code (e.g. `en`). Empty = auto-detect |
| `task` | string | `transcribe` (default) or `translate` |

**Response:**
```json
{
  "file": "interview.mp3",
  "duration_s": 127.4,
  "language": "en",
  "text": "Hello, welcome to the interview...",
  "segments": [
    {
      "id": 0,
      "text": "Hello, welcome to the interview.",
      "start": 0.0,
      "end": 2.84,
      "confidence": 0.9312,
      "words": [
        { "word": "Hello", "start": 0.0, "end": 0.42, "confidence": 0.9981 },
        ...
      ]
    }
  ],
  "processing_time_s": 8.23,
  "model": "tiny",
  "chunks_processed": 1,
  "audio_meta": {
    "extension": ".mp3",
    "size_mb": 2.14,
    "duration_s": 127.4,
    "format": "mp3",
    "sample_rate": "44100",
    "channels": "2",
    "bit_rate": "128000"
  }
}
```

### `GET /health`
Returns service status and whether the Whisper model is loaded.

### `GET /formats`
Returns list of all supported audio formats.

### `GET /docs`
Interactive Swagger UI.

---

## Engineering Decisions

### How different audio formats are handled
pydub (backed by ffmpeg) is used for format-agnostic audio loading. The format is detected from the file content, not just the extension. All audio is normalised to **16 kHz mono PCM** — the exact format Whisper requires — before transcription.

### How long audio files are handled
Files longer than **10 minutes** are automatically split into overlapping chunks:
- **Chunk size:** 10 minutes
- **Overlap:** 3 seconds (prevents words from being lost at boundaries)
- **Merge strategy:** Segments from chunk N+1 whose start time falls within the overlap window of chunk N are discarded (conservative deduplication)
- **Timestamp correction:** Each chunk's timestamps are shifted by the chunk's absolute offset

### Model selection
The Whisper model is configurable via the `WHISPER_MODEL` environment variable:

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| tiny | 75 MB | Fastest | Lower |
| base | 145 MB | Fast | Good |
| small | 483 MB | Moderate | Better |
| medium | 1.5 GB | Slow | High |
| large-v3 | 3 GB | Slowest | Best |

Default is `tiny` — significantly faster on CPUs with a minor drop in accuracy. Set `WHISPER_MODEL=base` or `medium` if you have a GPU or need higher accuracy.

### Production considerations
- **Async jobs:** For production, push transcription to Celery/RQ and poll a `/jobs/{id}` endpoint
- **Model caching:** Model is loaded once at startup and cached (singleton) to avoid repeated loading
- **Streaming uploads:** Files are streamed to disk in 1 MB chunks, never fully loaded into memory
- **GPU support:** Whisper uses CUDA automatically if available

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `tiny` | Whisper model size |

---

## Project Structure

```
audio-to-text-transcoder/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI application + routes
│   ├── audio_processor.py # Format detection, normalisation, chunking
│   ├── transcriber.py     # Whisper wrapper, chunk merge, timestamps
│   └── models.py          # Pydantic schemas
├── static/
│   ├── index.html         # Web UI
│   ├── style.css          # Premium dark-mode design
│   └── app.js             # Frontend logic
├── cli.py                 # Command-line interface
├── run.py                 # Server entry point
└── requirements.txt
```
