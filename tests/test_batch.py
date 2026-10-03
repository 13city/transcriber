from pathlib import Path

import pytest

from transcriber.batch import discover_jobs, select_jobs


def test_discover_prefers_raw_transcript_over_clean_and_audio(tmp_path: Path) -> None:
    lecture = tmp_path / "Lecture-One"
    lecture.mkdir()
    (lecture / "Lecture-One.flac").write_bytes(b"audio")
    (lecture / "Lecture-One.txt").write_text("clean-ish text", encoding="utf-8")
    raw = lecture / "Lecture-One.raw.txt"
    raw.write_text("[00:00:00 --> 00:00:01]\nraw text\n", encoding="utf-8")

    jobs = discover_jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].name == "Lecture-One"
    assert jobs[0].source == raw
    assert jobs[0].source_type == "transcript"


def test_discover_prefers_txt_over_flac_for_legacy_pairs(tmp_path: Path) -> None:
    (tmp_path / "Lecture-Two.flac").write_bytes(b"audio")
    transcript = tmp_path / "Lecture-Two.txt"
    transcript.write_text("[00:00:00 --> 00:00:01]\ntext\n", encoding="utf-8")

    jobs = discover_jobs(tmp_path)
    assert jobs[0].source == transcript


def test_discover_ignores_cleaned_output_directory(tmp_path: Path) -> None:
    (tmp_path / "Lecture.flac").write_bytes(b"audio")
    output = tmp_path / "Cleaned-Transcriptions"
    output.mkdir()
    (output / "Already-Clean.txt").write_text("clean", encoding="utf-8")

    jobs = discover_jobs(tmp_path, output)
    assert [job.name for job in jobs] == ["Lecture"]


def test_select_one_or_many(tmp_path: Path) -> None:
    (tmp_path / "Alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "Beta.txt").write_text("beta", encoding="utf-8")
    jobs = discover_jobs(tmp_path)

    assert [job.name for job in select_jobs(jobs, ["Alpha"])] == ["Alpha"]
    assert [job.name for job in select_jobs(jobs, ["Alpha", "Beta.txt"])] == [
        "Alpha",
        "Beta",
    ]


def test_select_accepts_human_title(tmp_path: Path) -> None:
    (tmp_path / "Assessing-Endurance-Performance.txt").write_text(
        "text", encoding="utf-8"
    )
    jobs = discover_jobs(tmp_path)
    selected = select_jobs(jobs, ["Assessing Endurance Performance"])
    assert selected[0].name == "Assessing-Endurance-Performance"


def test_select_missing_fails_clearly(tmp_path: Path) -> None:
    (tmp_path / "Alpha.txt").write_text("alpha", encoding="utf-8")
    jobs = discover_jobs(tmp_path)
    with pytest.raises(ValueError, match="Missing"):
        select_jobs(jobs, ["Missing"])
