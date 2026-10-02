from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from .text import Segment, seconds_to_timestamp


def transcribe_audio(
    audio_path: Path,
    output_dir: Path,
    base_name: str,
    model_name: str,
    device: str,
    compute_type: str,
) -> tuple[Path, Path, list[Segment], dict[str, Any]]:
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    generated, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=True,
    )

    segments: list[Segment] = []
    for item in generated:
        text = item.text.strip()
        if text:
            segments.append(
                Segment(start=float(item.start), end=float(item.end), text=text)
            )

    raw_path = output_dir / f"{base_name}.raw.txt"
    json_path = output_dir / f"{base_name}.segments.json"

    with raw_path.open("w", encoding="utf-8") as handle:
        for segment in segments:
            handle.write(
                f"[{seconds_to_timestamp(segment.start)} --> "
                f"{seconds_to_timestamp(segment.end)}]\n"
                f"{segment.text}\n\n"
            )

    payload = {
        "language": info.language,
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "model": model_name,
        "segments": [
            {"start": item.start, "end": item.end, "text": item.text}
            for item in segments
        ],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return raw_path, json_path, segments, payload
