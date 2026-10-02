from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    recordings_dir: Path
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    punctuation_model: str
    normalizer_model: str
    normalizer_device_map: str
    normalize_chunk_words: int

    @classmethod
    def from_env(cls) -> "Settings":
        base = Path(os.environ.get("TRANSCRIBER_RECORDINGS_DIR", "recordings")).expanduser()
        return cls(
            recordings_dir=base,
            whisper_model=os.environ.get("TRANSCRIBER_WHISPER_MODEL", "large-v3"),
            whisper_device=os.environ.get("TRANSCRIBER_WHISPER_DEVICE", "cpu"),
            whisper_compute_type=os.environ.get("TRANSCRIBER_WHISPER_COMPUTE", "int8"),
            punctuation_model=os.environ.get(
                "TRANSCRIBER_PUNCTUATION_MODEL",
                "oliverguhr/fullstop-punctuation-multilang-large",
            ),
            normalizer_model=os.environ.get(
                "TRANSCRIBER_NORMALIZER_MODEL", "Qwen/Qwen3-8B"
            ),
            normalizer_device_map=os.environ.get(
                "TRANSCRIBER_NORMALIZER_DEVICE_MAP", "auto"
            ),
            normalize_chunk_words=int(
                os.environ.get("TRANSCRIBER_NORMALIZE_CHUNK_WORDS", "650")
            ),
        )
