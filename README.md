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

The FullStop model is called directly through the current Transformers token-classification API. The project does not depend on the older `deepmultilingualpunctuation` wrapper, which uses a removed `grouped_entities` argument in some released versions.

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
python3 -m venv transcriber-env
source transcriber-env/bin/activate
pip install --upgrade pip
pip install -e .
```

Model weights are downloaded once from Hugging Face and cached locally. Lecture audio is processed locally.

## Record and process a lecture

```bash
transcriber record "Assessing Endurance Performance"
```

After you press `q` to stop FFmpeg, the tool automatically transcribes and normalizes the recording.

## Batch-clean an existing lecture collection

The batch command searches recursively and supports old `.flac` + `.txt` pairs as well as the newer per-lecture directory format.

For each lecture it prefers, in order:

1. `Lecture.raw.txt`
2. `Lecture.txt`
3. `Lecture.flac`

Existing transcripts are normalized directly. A FLAC is transcribed only when no transcript exists.

Cleaned transcripts are consolidated into:

```text
recordings/Cleaned-Transcriptions/
```

Inspect the queue:

```bash
transcriber batch ~/lecture-transcriber/recordings --list
```

Process all lectures sequentially:

```bash
transcriber batch ~/lecture-transcriber/recordings --all
```

Process one:

```bash
transcriber batch ~/lecture-transcriber/recordings \
  --select Assessing-Endurance-Performance
```

Process several:

```bash
transcriber batch ~/lecture-transcriber/recordings \
  --select Assessing-Endurance-Performance Threshold-Testing VO2-Max-Testing
```

The normalization models are loaded once and reused as the process moves from lecture to lecture. Existing cleaned transcripts are skipped automatically, so an interrupted batch can simply be rerun. Use `--force` only when you deliberately want to regenerate completed outputs.

## Process an existing recording

```bash
transcriber process recordings/Assessing-Endurance-Performance/Assessing-Endurance-Performance.flac
```

## Normalize an existing transcript

Both timestamped Faster Whisper output and ordinary plain-text transcripts are accepted:

```bash
transcriber normalize path/to/Assessing-Endurance-Performance.raw.txt
```

## Model and resource controls

```bash
export TRANSCRIBER_WHISPER_MODEL=large-v3
export TRANSCRIBER_WHISPER_DEVICE=cpu
export TRANSCRIBER_WHISPER_COMPUTE=int8
export TRANSCRIBER_PUNCTUATION_MODEL=oliverguhr/fullstop-punctuation-multilang-large
export TRANSCRIBER_NORMALIZER_MODEL=Qwen/Qwen3-8B
export TRANSCRIBER_NORMALIZER_DEVICE_MAP=auto
export TRANSCRIBER_NORMALIZE_CHUNK_WORDS=650
```

To see the active configuration:

```bash
transcriber models
```

## Privacy and course access

This project is intended for material you are authorized to access and for personal/local study workflows. It does not bypass DRM or LMS access controls. The recorder captures audio that the operating system is already playing. Do not commit course recordings or transcripts to this repository unless you have permission to redistribute them.
