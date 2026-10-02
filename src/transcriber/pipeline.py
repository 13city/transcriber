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


def normalize_raw_transcript(
    raw_path: Path,
    output_path: Path,
    settings: Settings,
    use_punctuation_model: bool = True,
    use_llm: bool = True,
) -> Path:
    segments = parse_timestamped_transcript(raw_path)
    if not segments:
        raise ValueError(f"No timestamped segments found in {raw_path}")

    segments = deduplicate_segments(segments)
    text = segments_to_plain_text(segments)

    punctuation_used = False
    if use_punctuation_model and sentence_punctuation_density(text) < 0.012:
        punctuator = FullStopPunctuator(settings.punctuation_model)
        text = punctuator.restore(text)
        punctuation_used = True

    audit: dict[str, object] = {
        "source": str(raw_path),
        "output": str(output_path),
        "punctuation_model": settings.punctuation_model if punctuation_used else None,
        "normalizer_model": settings.normalizer_model if use_llm else None,
        "input_words_after_dedupe": len(text.split()),
        "warnings": [],
    }

    if use_llm:
        result = QwenNormalizer(
            settings.normalizer_model,
            device_map=settings.normalizer_device_map,
            chunk_words=settings.normalize_chunk_words,
        ).normalize(text)
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
