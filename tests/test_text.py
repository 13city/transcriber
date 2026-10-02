from pathlib import Path

from transcriber.text import (
    Segment,
    deduplicate_segments,
    parse_timestamped_transcript,
    slugify_title,
)


def test_slugify_title() -> None:
    assert (
        slugify_title("Assessing Endurance Performance")
        == "Assessing-Endurance-Performance"
    )


def test_parse_timestamped_transcript(tmp_path: Path) -> None:
    path = tmp_path / "sample.raw.txt"
    path.write_text(
        "[00:00:02 --> 00:00:06]\nWelcome to the course.\n\n"
        "[00:00:06 --> 00:00:10]\nThis is the next sentence.\n",
        encoding="utf-8",
    )
    segments = parse_timestamped_transcript(path)
    assert len(segments) == 2
    assert segments[1].text == "This is the next sentence."


def test_deduplicate_exact_adjacent_asr_repeat() -> None:
    segments = [
        Segment(10, 12, "Both have been correlated to average power output."),
        Segment(12, 12.5, "Both have been correlated to average power output."),
    ]
    cleaned = deduplicate_segments(segments)
    assert len(cleaned) == 1


def test_trim_boundary_overlap() -> None:
    segments = [
        Segment(0, 4, "We will discuss threshold intensity and cycling economy"),
        Segment(4, 8, "threshold intensity and cycling economy in this lecture"),
    ]
    cleaned = deduplicate_segments(segments)
    assert cleaned[1].text == "in this lecture"
