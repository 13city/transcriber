from __future__ import annotations

import json
from pathlib import Path

from .asr import transcribe_audio
from .config import Settings
from .normalize import QwenNormalizer
from .punctuation import FullStopPunctuator
from .text import (
    deduplicate_segments,
    parse_timestamped_transcript,
    segments_to_plain_text,
    sentence_punctuation_density,
)


def _load_source_text(raw_path: Path) -> tuple[str, str]:
    """Load a timestamped Whisper transcript or fall back to plain text."""
    segments = parse_timestamped_transcript(raw_path)
    if segments:
        segments = deduplicate_segments(segments)
        return segments_to_plain_text(segments), "timestamped"

    text = raw_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Transcript is empty: {raw_path}")
    return text, "plain-text"


def normalize_raw_transcript(
    raw_path: Path,
    output_path: Path,
    settings: Settings,
    use_punctuation_model: bool = True,
    use_llm: bool = True,
    punctuator: FullStopPunctuator | None = None,
    normalizer: QwenNormalizer | None = None,
) -> Path:
    raw_path = raw_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text, source_mode = _load_source_text(raw_path)

    punctuation_used = False
    if use_punctuation_model and sentence_punctuation_density(text) < 0.012:
        active_punctuator = punctuator or FullStopPunctuator(settings.punctuation_model)
        text = active_punctuator.restore(text)
        punctuation_used = True

    audit: dict[str, object] = {
        "source": str(raw_path),
        "source_mode": source_mode,
        "output": str(output_path),
        "punctuation_model": settings.punctuation_model if punctuation_used else None,
        "normalizer_model": settings.normalizer_model if use_llm else None,
        "input_words_after_dedupe": len(text.split()),
        "warnings": [],
    }

    if use_llm:
        active_normalizer = normalizer or QwenNormalizer(
            settings.normalizer_model,
            device_map=settings.normalizer_device_map,
            chunk_words=settings.normalize_chunk_words,
        )
        result = active_normalizer.normalize(text)
        text = result.text
        audit.update(
            {
                "normalized_input_words": result.input_words,
                "normalized_output_words": result.output_words,
                "length_ratio": result.length_ratio,
                "warnings": result.warnings,
            }
        )
    else:
        text = text.strip()

    output_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    audit_path = output_path.with_suffix(".audit.json")
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def process_audio(
    audio_path: Path,
    settings: Settings,
    use_punctuation_model: bool = True,
    use_llm: bool = True,
) -> tuple[Path, Path, Path]:
    audio_path = audio_path.expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    output_dir = audio_path.parent
    base_name = audio_path.stem
    raw_path, segments_path, _segments, _metadata = transcribe_audio(
        audio_path=audio_path,
        output_dir=output_dir,
        base_name=base_name,
        model_name=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    clean_path = output_dir / f"{base_name}.txt"
    normalize_raw_transcript(
        raw_path,
        clean_path,
        settings=settings,
        use_punctuation_model=use_punctuation_model,
        use_llm=use_llm,
    )
    return raw_path, clean_path, segments_path
