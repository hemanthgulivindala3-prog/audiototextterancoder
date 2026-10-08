#!/usr/bin/env python3
"""
cli.py
------
Command-line interface for the transcription pipeline.
Allows transcribing audio files without starting the web server.

Usage:
  python cli.py audio.mp3
  python cli.py audio.wav --model medium --language en
  python cli.py audio.m4a --task translate --output result.json
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="AudioScript CLI — Transcribe audio files using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", help="Path to the audio file")
    parser.add_argument(
        "--model", "-m",
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Language code (e.g. 'en', 'fr'). Auto-detect if not set.",
    )
    parser.add_argument(
        "--task", "-t",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="'transcribe' keeps original language; 'translate' → English",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file path. Prints to stdout if not set.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Print only the transcript text, not the full JSON",
    )
    args = parser.parse_args()

    # Set model env var so transcriber picks it up
    os.environ["WHISPER_MODEL"] = args.model

    # Import here so --help works without torch/whisper installed
    from app.transcriber import transcribe_audio_file

    print(f"[AudioScript CLI] Transcribing: {args.file}", file=sys.stderr)
    print(f"[AudioScript CLI] Model: {args.model}, Language: {args.language or 'auto'}, Task: {args.task}", file=sys.stderr)

    result = transcribe_audio_file(
        file_path=args.file,
        language=args.language,
        task=args.task,
    )

    print(f"\n✓ Done in {result['processing_time_s']}s | Language: {result['language']} | Segments: {len(result['segments'])}", file=sys.stderr)

    if args.text_only:
        output = result["text"]
    else:
        output = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(output, encoding="utf-8")
        print(f"✓ Output saved to: {out_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
