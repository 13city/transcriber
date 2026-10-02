from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings
from .pipeline import normalize_raw_transcript, process_audio
from .record import detect_monitor_source, record_lecture


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcriber",
        description="Local-first lecture capture, transcription, and normalization.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="Record system audio to a lecture folder")
    record.add_argument(
        "title",
        help="Human lecture title; filenames are hyphenated automatically",
    )
    record.add_argument(
        "--source",
        help="PipeWire/PulseAudio monitor source; auto-detected by default",
    )
    record.add_argument(
        "--no-process",
        action="store_true",
        help="Record only; do not transcribe and normalize after recording",
    )
    record.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Qwen normalization",
    )
    record.add_argument(
        "--no-punctuation-model",
        action="store_true",
        help="Disable FullStop punctuation restoration",
    )

    process = sub.add_parser(
        "process",
        help="Transcribe and normalize an existing audio file",
    )
    process.add_argument("audio", type=Path)
    process.add_argument("--no-llm", action="store_true")
    process.add_argument("--no-punctuation-model", action="store_true")

    normalize = sub.add_parser(
        "normalize",
        help="Normalize an existing timestamped raw transcript",
    )
    normalize.add_argument("raw_transcript", type=Path)
    normalize.add_argument("--output", type=Path)
    normalize.add_argument("--no-llm", action="store_true")
    normalize.add_argument("--no-punctuation-model", action="store_true")

    sub.add_parser(
        "audio-source",
        help="Print the auto-detected system-output monitor source",
    )
    sub.add_parser("models", help="Print configured local model IDs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings.from_env()

    try:
        if args.command == "audio-source":
            print(detect_monitor_source())
            return 0

        if args.command == "models":
            print(f"Whisper:     {settings.whisper_model}")
            print(f"Punctuation: {settings.punctuation_model}")
            print(f"Normalizer:  {settings.normalizer_model}")
            return 0

        if args.command == "record":
            audio = record_lecture(
                args.title,
                settings.recordings_dir,
                args.source,
            )
            if args.no_process:
                print(f"\nRecording saved: {audio}")
                return 0
            raw, clean, segments = process_audio(
                audio,
                settings,
                use_punctuation_model=not args.no_punctuation_model,
                use_llm=not args.no_llm,
            )
            print(f"\nRaw transcript:   {raw}")
            print(f"Clean transcript: {clean}")
            print(f"Segments JSON:    {segments}")
            return 0

        if args.command == "process":
            raw, clean, segments = process_audio(
                args.audio,
                settings,
                use_punctuation_model=not args.no_punctuation_model,
                use_llm=not args.no_llm,
            )
            print(f"Raw transcript:   {raw}")
            print(f"Clean transcript: {clean}")
            print(f"Segments JSON:    {segments}")
            return 0

        if args.command == "normalize":
            raw = args.raw_transcript.expanduser().resolve()
            output = args.output or raw.with_name(
                raw.name.replace(".raw.txt", ".txt")
            )
            normalize_raw_transcript(
                raw,
                output,
                settings,
                use_punctuation_model=not args.no_punctuation_model,
                use_llm=not args.no_llm,
            )
            print(f"Clean transcript: {output}")
            return 0

    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
