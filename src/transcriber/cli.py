from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .batch import discover_jobs, run_batch, select_jobs
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
        help="Normalize an existing timestamped or plain-text transcript",
    )
    normalize.add_argument("raw_transcript", type=Path)
    normalize.add_argument("--output", type=Path)
    normalize.add_argument("--no-llm", action="store_true")
    normalize.add_argument("--no-punctuation-model", action="store_true")

    batch = sub.add_parser(
        "batch",
        help="Sequentially clean one, several, or all transcripts in a directory tree",
    )
    batch.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("recordings"),
        help="Directory containing .txt/.raw.txt/.flac lecture files (default: recordings)",
    )
    mode = batch.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "--all",
        action="store_true",
        help="Process every discovered lecture",
    )
    mode.add_argument(
        "--select",
        nargs="+",
        metavar="LECTURE",
        help="Process only the named lecture(s); extensions are optional",
    )
    batch.add_argument(
        "--output-dir",
        type=Path,
        help="Destination for all cleaned transcripts (default: INPUT/Cleaned-Transcriptions)",
    )
    batch.add_argument(
        "--list",
        action="store_true",
        help="List discovered lectures without loading models or processing files",
    )
    batch.add_argument(
        "--force",
        action="store_true",
        help="Replace cleaned transcripts that already exist",
    )
    batch.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one lecture fails instead of continuing",
    )
    batch.add_argument("--no-llm", action="store_true")
    batch.add_argument("--no-punctuation-model", action="store_true")

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

        if args.command == "batch":
            input_dir = args.input_dir.expanduser().resolve()
            output_dir = (
                args.output_dir.expanduser().resolve()
                if args.output_dir
                else input_dir / "Cleaned-Transcriptions"
            )

            if args.list:
                jobs = select_jobs(
                    discover_jobs(input_dir, output_dir),
                    args.select,
                )
                if not jobs:
                    print("No lectures found.")
                    return 0
                for job in jobs:
                    print(f"{job.name}\t{job.source_type}\t{job.source}")
                return 0

            if not args.all and not args.select:
                print(
                    "error: choose --all or --select LECTURE [LECTURE ...]. "
                    "Use --list to inspect discovered lectures.",
                    file=sys.stderr,
                )
                return 2

            summary = run_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                settings=settings,
                selections=args.select,
                force=args.force,
                stop_on_error=args.stop_on_error,
                use_punctuation_model=not args.no_punctuation_model,
                use_llm=not args.no_llm,
            )
            print(
                "\nBatch complete: "
                f"{summary.completed} completed, "
                f"{summary.skipped} skipped, "
                f"{summary.failed} failed."
            )
            return 1 if summary.failed else 0

    except (
        FileNotFoundError,
        FileExistsError,
        NotADirectoryError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
