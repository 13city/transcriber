from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

TIMESTAMP_RE = re.compile(
    r"^\[(?P<start>\d{2}:\d{2}:\d{2})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2})\]\s*$"
)


@dataclass
class Segment:
    start: float
    end: float
    text: str


def slugify_title(title: str) -> str:
    value = unicodedata.normalize("NFKC", title).strip()
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    value = re.sub(r"[-\s]+", "-", value)
    return value.strip("-")


def seconds_to_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def timestamp_to_seconds(value: str) -> float:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return float(hours * 3600 + minutes * 60 + seconds)


def parse_timestamped_transcript(path: Path) -> list[Segment]:
    lines = path.read_text(encoding="utf-8").splitlines()
    segments: list[Segment] = []
    index = 0
    while index < len(lines):
        match = TIMESTAMP_RE.match(lines[index].strip())
        if not match:
            index += 1
            continue
        start = timestamp_to_seconds(match.group("start"))
        end = timestamp_to_seconds(match.group("end"))
        index += 1
        text_lines: list[str] = []
        while index < len(lines) and not TIMESTAMP_RE.match(lines[index].strip()):
            line = lines[index].strip()
            if line:
                text_lines.append(line)
            index += 1
        text = " ".join(text_lines).strip()
        if text:
            segments.append(Segment(start=start, end=end, text=text))
    return segments


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower(), flags=re.UNICODE)


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(
        None,
        " ".join(_normalized_words(left)),
        " ".join(_normalized_words(right)),
    ).ratio()


def _trim_boundary_overlap(previous: Segment, current: Segment) -> Segment:
    if current.start - previous.end > 1.5:
        return current

    prev_words = previous.text.split()
    curr_words = current.text.split()
    max_overlap = min(25, len(prev_words), len(curr_words))
    for size in range(max_overlap, 4, -1):
        left = " ".join(prev_words[-size:])
        right = " ".join(curr_words[:size])
        if _similarity(left, right) >= 0.96:
            return Segment(current.start, current.end, " ".join(curr_words[size:]).strip())
    return current


def deduplicate_segments(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return []

    result: list[Segment] = []
    for segment in segments:
        if not segment.text.strip():
            continue
        if result:
            previous = result[-1]
            close_in_time = segment.start - previous.end <= 2.0
            if close_in_time and _similarity(previous.text, segment.text) >= 0.97:
                continue
            segment = _trim_boundary_overlap(previous, segment)
            if not segment.text:
                continue
        result.append(segment)
    return result


def segments_to_plain_text(segments: list[Segment]) -> str:
    return re.sub(
        r"\s+",
        " ",
        " ".join(segment.text.strip() for segment in segments),
    ).strip()


def sentence_punctuation_density(text: str) -> float:
    words = max(1, len(text.split()))
    boundaries = len(re.findall(r"[.!?](?:\s|$)", text))
    return boundaries / words


def split_words_for_llm(text: str, target_words: int) -> list[str]:
    """Split long text near sentence boundaries without dropping content."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) == 1:
        words = text.split()
        return [
            " ".join(words[i : i + target_words])
            for i in range(0, len(words), target_words)
        ]

    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        size = len(sentence.split())
        if current and count + size > target_words:
            chunks.append(" ".join(current).strip())
            current = []
            count = 0
        current.append(sentence)
        count += size
    if current:
        chunks.append(" ".join(current).strip())
    return chunks
