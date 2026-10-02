from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .asr import transcribe_audio
from .config import Settings
from .normalize import QwenNormalizer
from .pipeline import normalize_raw_transcript
from .punctuation import FullStopPunctuator
from .text import slugify_title


@dataclass(frozen=True)
class BatchJob:
    name: str
    source: Path
    source_type: str


@dataclass
class BatchSummary:
    selected: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0


def _base_name_for(path: Path) -> str:
    if path.name.endswith(".raw.txt"):
        return path.name[: -len(".raw.txt")]
    return path.stem


def _candidate_priority(path: Path) -> int:
    if path.name.endswith(".raw.txt"):
        return 0
    if path.suffix.lower() == ".txt":
        return 1
    if path.suffix.lower() == ".flac":
        return 2
    return 99


def discover_jobs(input_dir: Path, output_dir: Path | None = None) -> list[BatchJob]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    resolved_output = output_dir.expanduser().resolve() if output_dir else None
    candidates: dict[str, Path] = {}

    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if resolved_output and (path == resolved_output or resolved_output in path.parents):
            continue
        if path.name.endswith(".audit.json") or path.name.endswith(".segments.json"):
            continue
        if path.suffix.lower() not in {".txt", ".flac"}:
            continue

        name = _base_name_for(path)
        current = candidates.get(name)
        if current is None or _candidate_priority(path) < _candidate_priority(current):
            candidates[name] = path

    jobs = [
        BatchJob(
            name=name,
            source=path,
            source_type="audio" if path.suffix.lower() == ".flac" else "transcript",
        )
        for name, path in candidates.items()
    ]
    return sorted(jobs, key=lambda item: item.name.casefold())


def select_jobs(jobs: list[BatchJob], selections: list[str] | None) -> list[BatchJob]:
    if not selections:
        return jobs

    by_name = {job.name: job for job in jobs}
    folded = {job.name.casefold(): job for job in jobs}
    chosen: list[BatchJob] = []
    missing: list[str] = []

    for selection in selections:
        candidate = selection
        for suffix in (".raw.txt", ".txt", ".flac"):
            if candidate.endswith(suffix):
                candidate = candidate[: -len(suffix)]
                break

        job = by_name.get(candidate)
        if job is None:
            job = folded.get(candidate.casefold())
        if job is None:
            slug = slugify_title(candidate)
            job = by_name.get(slug) or folded.get(slug.casefold())

        if job is None:
            missing.append(selection)
        elif job not in chosen:
            chosen.append(job)

    if missing:
        raise ValueError(
            "No matching lecture found for: " + ", ".join(missing)
        )
    return chosen


def _raw_from_audio(job: BatchJob, settings: Settings) -> Path:
    output_dir = job.source.parent
    raw_path, _segments_path, _segments, _metadata = transcribe_audio(
        audio_path=job.source,
        output_dir=output_dir,
        base_name=job.name,
        model_name=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    return raw_path


def run_batch(
    input_dir: Path,
    output_dir: Path,
    settings: Settings,
    selections: list[str] | None = None,
    force: bool = False,
    stop_on_error: bool = False,
    use_punctuation_model: bool = True,
    use_llm: bool = True,
) -> BatchSummary:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = select_jobs(discover_jobs(input_dir, output_dir), selections)
    summary = BatchSummary(selected=len(jobs))
    if not jobs:
        return summary

    print(f"Found {len(jobs)} lecture(s).")
    print(f"Cleaned transcripts: {output_dir}")

    # Load expensive models once and reuse them for every lecture.
    normalizer = (
        QwenNormalizer(
            settings.normalizer_model,
            device_map=settings.normalizer_device_map,
            chunk_words=settings.normalize_chunk_words,
        )
        if use_llm
        else None
    )
    punctuator = (
        FullStopPunctuator(settings.punctuation_model)
        if use_punctuation_model
        else None
    )

    for index, job in enumerate(jobs, start=1):
        destination = output_dir / f"{job.name}.txt"
        print(f"\n[{index}/{len(jobs)}] {job.name}")
        print(f"Source: {job.source}")

        if destination.exists() and not force:
            print("Skipped: cleaned transcript already exists. Use --force to replace it.")
            summary.skipped += 1
            continue

        try:
            source_transcript = (
                _raw_from_audio(job, settings)
                if job.source_type == "audio"
                else job.source
            )
            normalize_raw_transcript(
                source_transcript,
                destination,
                settings=settings,
                use_punctuation_model=use_punctuation_model,
                use_llm=use_llm,
                punctuator=punctuator,
                normalizer=normalizer,
            )
            print(f"Saved: {destination}")
            summary.completed += 1
        except Exception as exc:
            summary.failed += 1
            print(f"FAILED: {exc}")
            if stop_on_error:
                raise

    return summary
