# Transcriber

A local-first tool for turning paid/authorized lecture playback into organized, useful transcripts without uploading the lecture audio to a transcription service.

The pipeline is deliberately provenance-preserving:

```text
system audio
  -> FLAC recording
  -> Faster Whisper raw transcript + timestamped segments
  -> deterministic boundary de-duplication
  -> FullStop punctuation restoration when punctuation is unusually sparse
  -> conservative Qwen3 normalization into sentences and paragraphs
  -> clean transcript + audit metadata
```

The raw Whisper transcript is always retained. The normalization model never replaces the source artifact.

## Models

Defaults:

- ASR: `large-v3` through `faster-whisper`
- punctuation: `oliverguhr/fullstop-punctuation-multilang-large`
- transcript normalization: `Qwen/Qwen3-8B` through Hugging Face Transformers

The punctuation model is invoked only when the transcript is unusually sparse in sentence punctuation. Faster Whisper often already supplies useful punctuation, and avoiding unnecessary restoration protects technical notation, decimals, and abbreviations.

Qwen runs in non-thinking mode with deterministic decoding and a restrictive editor prompt. It is told not to summarize, add outside knowledge, or silently guess uncertain terminology. An audit JSON flags unusually large changes in transcript length.

## Debian prerequisites

```bash
sudo apt update
sudo apt install -y ffmpeg python3 python3-venv python3-pip git pulseaudio-utils
```

## Install

```bash
git clone https://github.com/13city/transcriber.git
cd transcriber
python3 -m venv lecture-transcription-env
source lecture-transcription-env/bin/activate
pip install --upgrade pip
pip install -e .
```

Model weights are downloaded once from Hugging Face and cached locally. Lecture audio is processed locally.

## Record and process a lecture

The title can contain normal spaces. The tool creates a filesystem-safe hyphenated directory and matching filenames automatically:

```bash
transcriber record "Assessing Endurance Performance"
```

After you press `q` to stop FFmpeg, the tool automatically transcribes and normalizes the recording.

Result:

```text
recordings/
└── Assessing-Endurance-Performance/
    ├── Assessing-Endurance-Performance.flac
    ├── Assessing-Endurance-Performance.raw.txt
    ├── Assessing-Endurance-Performance.segments.json
    ├── Assessing-Endurance-Performance.txt
    └── Assessing-Endurance-Performance.audit.json
```

The `recordings/` directory and lecture media/transcript artifacts are gitignored by default.

## Batch-clean an existing lecture collection

The batch command is designed for a directory containing many old `.flac` + `.txt` pairs or the newer per-lecture directory format. It searches recursively.

For each lecture it prefers, in order:

1. `Lecture.raw.txt`
2. `Lecture.txt`
3. `Lecture.flac`

That means an existing transcript is normalized directly instead of wasting time transcribing the FLAC again. If only a FLAC exists, Faster Whisper is run first.

All cleaned transcripts are written into one directory:

```text
recordings/
└── Cleaned-Transcriptions/
    ├── Assessing-Endurance-Performance.txt
    ├── Assessing-Endurance-Performance.audit.json
    ├── Threshold-Testing.txt
    ├── Threshold-Testing.audit.json
    └── ...
```

Original FLAC and transcript files are left untouched.

First inspect what will be processed:

```bash
transcriber batch --list
```

Process every discovered lecture sequentially:

```bash
transcriber batch --all
```

Process one:

```bash
transcriber batch --select Assessing-Endurance-Performance
```

Process several in one run:

```bash
transcriber batch --select Assessing-Endurance-Performance Threshold-Testing VO2-Max-Testing
```

If your source collection is somewhere other than `recordings/`:

```bash
transcriber batch /path/to/lecture-folder --all
```

Choose a different output directory:

```bash
transcriber batch /path/to/lecture-folder --all \
  --output-dir /path/to/Cleaned-Transcriptions
```

The Qwen and punctuation models are loaded once and reused as the program moves from lecture to lecture. Processing is strictly sequential, so only one lecture is normalized at a time.

If the batch is interrupted, rerun the same command. Existing cleaned transcripts are skipped automatically. To deliberately replace them:

```bash
transcriber batch --all --force
```

A failed lecture is reported and the batch continues to the next one. Use `--stop-on-error` if you instead want the first failure to halt the run.

## Process an existing recording

```bash
transcriber process recordings/Assessing-Endurance-Performance/Assessing-Endurance-Performance.flac
```

## Normalize an existing transcript

Timestamped Faster Whisper output and ordinary plain-text transcripts are both accepted:

```bash
transcriber normalize path/to/Assessing-Endurance-Performance.raw.txt
```

## Model and resource controls

Environment variables:

```bash
export TRANSCRIBER_WHISPER_MODEL=large-v3
export TRANSCRIBER_WHISPER_DEVICE=cpu
export TRANSCRIBER_WHISPER_COMPUTE=int8
export TRANSCRIBER_PUNCTUATION_MODEL=oliverguhr/fullstop-punctuation-multilang-large
export TRANSCRIBER_NORMALIZER_MODEL=Qwen/Qwen3-8B
export TRANSCRIBER_NORMALIZER_DEVICE_MAP=auto
export TRANSCRIBER_NORMALIZE_CHUNK_WORDS=650
```

To see the active model configuration:

```bash
transcriber models
```

## Audio source detection

The recorder normally discovers the default sink monitor automatically:

```bash
transcriber audio-source
```

If multiple monitor sources exist, specify one explicitly:

```bash
transcriber record "Lecture Title" --source alsa_output.pci-0000_00_1f.3.analog-stereo.monitor
```

## Conservative fallbacks

Skip Qwen normalization while still creating the raw transcript:

```bash
transcriber process lecture.flac --no-llm
```

Disable FullStop punctuation restoration:

```bash
transcriber process lecture.flac --no-punctuation-model
```

## Privacy and course access

This project is intended for material you are authorized to access and for personal/local study workflows. It does not bypass DRM or LMS access controls. The recorder captures audio that the operating system is already playing. Do not commit course recordings or transcripts to this repository unless you have permission to redistribute them.
