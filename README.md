# translate

Near-real-time Japanese → English transcription and translation of system audio (Teams calls, YouTube, anything playing through your speakers). Fully local — no API keys, no cloud, no data leaves the machine. CPU-primary with optional GPU acceleration.

## Aim

The goal is a frictionless always-on overlay: you hear Japanese, you read English. Latency is kept low enough to follow a live conversation (~2–5 s end-to-end on a modern CPU) without sacrificing translation quality. Default model choices favour purpose-built translation networks over general LLMs for better JA→EN fidelity.

## How it works

Two pipeline modes, two ASR backends.

**Default (ASR + NLLB MT)**
```
System audio
    │
    ▼
Audio capture (WASAPI loopback / PulseAudio monitor)
    │  raw PCM frames @ 16 kHz mono
    ▼
VAD — Silero VAD (ONNX, CPU)
    │  finalizes a chunk on silence gap or hard cap
    ▼
ASR — Whisper (ctranslate2 int8 OR OpenVINO int8 on NPU)
    │  emits Japanese transcript; shown in overlay immediately
    ▼
MT — NLLB-200 distilled 1.3B (transformers)
    │  translates JA → EN (sentence-level)
    ▼
Overlay — English translation
```

**Bilingual mode (`--bilingual`)** — single Whisper model, lower latency
```
System audio → VAD → Whisper (task=translate) → Overlay — English
```
Whisper translates JA→EN directly in one pass. No MT loaded. Works with either ASR backend.

Each stage runs on its own thread. The ASR and MT threads pull from bounded queues, so back-pressure is handled automatically and the UI stays responsive even if MT falls behind.

### ASR backends

Both backends implement the same interface (`transcribe`/`translate`); switch with `--asr-backend`:

- **`ctranslate2`** (default) — `faster-whisper` int8 on CPU or CUDA. Mature, fast on x86 and NVIDIA GPUs.
- **`openvino`** — `optimum-intel` + OpenVINO runtime on Intel **NPU**, iGPU, or CPU. Built for Core Ultra laptops without a discrete GPU. Requires the optional `openvino` extra (`uv sync --extra openvino`).

### Stage details

| Stage | Default model | Size | Notes |
|---|---|---|---|
| **Audio** | system loopback | — | WASAPI on Windows, PulseAudio/PipeWire monitor on Linux |
| **VAD** | Silero VAD (ONNX) | ~2 MB | 512-sample windows @ 16 kHz; speech threshold 0.5 |
| **ASR (ctranslate2)** | `Systran/faster-whisper-small` (int8) | ~240 MB | greedy decode; upgrade to `kotoba-whisper-v2.0-faster` for GPU |
| **ASR (ctranslate2, bilingual)** | `kotoba-tech/kotoba-whisper-bilingual-v1.0-faster` | ~1.5 GB | Whisper `task=translate` |
| **ASR (openvino)** | `OpenVINO/whisper-small-int8-ov` | ~250 MB | prequantized OpenVINO IR; runs on NPU/GPU/CPU via OpenVINO |
| **MT** | `facebook/nllb-200-distilled-1.3B` | ~5 GB fp32 / ~2.5 GB fp16 | translation-specialist seq2seq; 4-beam search; skipped under `--bilingual` |
| **UI** | PySide6 overlay | — | always-on-top, dual-pane, transparent background |

### Why threads, not async?

`faster-whisper`, `torch`, and `transformers` all release the GIL during their native compute. Threading gives real parallelism across the three stages without the complexity of multiprocessing or the overhead of asyncio bridging C extensions.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for environment management
- **Linux**: PulseAudio or PipeWire-pulse (default on Ubuntu 22.04+, Fedora 35+, modern Arch)
- **Windows**: nothing extra; WASAPI loopback works out of the box

## Setup

```bash
uv sync                                   # base install (ctranslate2 backend)
uv sync --extra openvino                  # add OpenVINO/NPU support

uv run translate                          # default: faster-whisper + NLLB-200
uv run translate --bilingual              # Whisper translate only, no MT
uv run translate --device cuda            # NVIDIA GPU for ctranslate2 ASR + torch MT
uv run translate --asr-backend openvino                   # OpenVINO AUTO device
uv run translate --asr-backend openvino --ov-device NPU   # Intel NPU
```

First run downloads model weights to the HuggingFace cache (`~/.cache/huggingface`). Subsequent starts load from disk in a few seconds.

### Intel NPU laptops (no discrete GPU)

On Core Ultra (Meteor Lake / Lunar Lake / Arrow Lake) install the extra:

```bash
uv sync --extra openvino
uv run translate --asr-backend openvino --ov-device NPU
```

Defaults to `OpenVINO/whisper-small-int8-ov`. Override with `--asr-model OpenVINO/whisper-large-v3-int4-ov` for higher quality, or `--ov-device AUTO` to let OpenVINO pick across NPU/iGPU/CPU. The MT stage (NLLB) still runs on torch CPU/CUDA — pair with `--bilingual` if MT latency is the bottleneck on NPU-only hardware.

## Configuration

All tuneable parameters live in [src/translate/config.py](src/translate/config.py). Key knobs:

- `ASRConfig.model` / `ASRConfig.device` — ctranslate2 ASR model and device (`cpu` / `cuda` / `auto`)
- `ASRConfig.bilingual_model` — used when `--bilingual --asr-backend ctranslate2`
- `OpenVINOASRConfig.model` / `OpenVINOASRConfig.device` — OpenVINO ASR model and device (`CPU` / `GPU` / `NPU` / `AUTO`)
- `MTConfig.model_repo` / `MTConfig.device` / `MTConfig.num_beams` — NLLB model, device, and beam width
- `VADConfig.silence_ms` — silence gap before a chunk is finalized; lower = more responsive but shorter context per chunk

## Status

Core pipeline implemented and wired. Models load and run; the overlay displays live output. Further work: hotkey toggling, per-language confidence filtering, streaming partial transcripts.
