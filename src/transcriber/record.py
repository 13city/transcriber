from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .text import slugify_title


def _run_capture(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def detect_monitor_source() -> str:
    if shutil.which("pactl") is None:
        raise RuntimeError(
            "pactl is required to auto-detect the PipeWire/PulseAudio monitor source"
        )

    sink = _run_capture(["pactl", "get-default-sink"])
    candidate = f"{sink}.monitor"
    sources = _run_capture(["pactl", "list", "short", "sources"])
    if candidate in sources:
        return candidate

    monitors = []
    for line in sources.splitlines():
        fields = line.split("\t")
        if len(fields) >= 2 and fields[1].endswith(".monitor"):
            monitors.append(fields[1])
    if len(monitors) == 1:
        return monitors[0]
    if not monitors:
        raise RuntimeError("No PipeWire/PulseAudio monitor source found")
    raise RuntimeError(
        "Multiple monitor sources found; pass --source explicitly:\n"
        + "\n".join(monitors)
    )


def record_lecture(
    title: str,
    recordings_dir: Path,
    source: str | None = None,
) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for recording")

    slug = slugify_title(title)
    if not slug:
        raise ValueError("Lecture title produced an empty filename")

    lecture_dir = recordings_dir.expanduser() / slug
    lecture_dir.mkdir(parents=True, exist_ok=True)
    audio_path = lecture_dir / f"{slug}.flac"
    if audio_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing recording: {audio_path}. "
            "Choose a different title."
        )

    monitor = source or detect_monitor_source()
    command = [
        "ffmpeg",
        "-hide_banner",
        "-f",
        "pulse",
        "-i",
        monitor,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "flac",
        str(audio_path),
    ]

    print(f"Lecture: {title}")
    print(f"Audio source: {monitor}")
    print(f"Recording to: {audio_path}")
    print("Start the lecture. Press q in this terminal when it ends.\n")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with status {completed.returncode}")
    return audio_path
