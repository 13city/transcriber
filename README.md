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
sudo apt install -y ffmpeg python3 python3-venv python3-pip git
```

The recorder uses the PulseAudio compatibility interface exposed by PipeWire. `pactl` is normally provided by `pulseaudio-utils`:

```bash
sudo apt install -y pulseaudio-utils
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

## Process an existing recording

```bash
transcriber process recordings/Assessing-Endurance-Performance/Assessing-Endurance-Performance.flac
```

## Normalize a transcript already produced by the old script

The input must use the existing timestamp format:

```text
[00:00:02 --> 00:00:06]
Welcome to Module 2 of the Cycling Science course.
```

Run:

```bash
transcriber normalize path/to/Assessing-Endurance-Performance.raw.txt
```

To preserve an old transcript before normalization, rename it to `.raw.txt` first.

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

`Qwen/Qwen3-8B` is the quality-oriented default. On a CPU-only machine with limited RAM, a smaller Qwen3 instruction model can be selected without changing the pipeline, for example:

```bash
export TRANSCRIBER_NORMALIZER_MODEL=Qwen/Qwen3-4B
```

Then run the same commands normally.

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
