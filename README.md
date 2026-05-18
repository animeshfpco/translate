# translate

Near-real-time Japanese → English transcription and translation of system audio (Teams calls, YouTube, anything playing through your speakers). Fully local — no API keys, no cloud, no data leaves the machine. CPU-primary with optional GPU acceleration.

## Aim

The goal is a frictionless always-on overlay: you hear Japanese, you read English. Latency is kept low enough to follow a live conversation (~2–5 s end-to-end on a modern CPU) without sacrificing translation quality. Every model choice favours small quantized variants that run well without a GPU.

## How it works

```
System audio
    │
    ▼
Audio capture (WASAPI loopback / PulseAudio monitor)
    │  raw PCM frames @ 16 kHz mono
    ▼
VAD — Silero VAD (ONNX, CPU)
    │  finalizes a chunk on ~300 ms silence or 8 s hard cap
    ▼
ASR — faster-whisper (int8)
    │  emits Japanese transcript; shown in overlay immediately
    ▼
MT — llama-cpp-python (Q4_K_M GGUF)
    │  translates JA → EN with rolling 4-turn context window
    ▼
Overlay — PySide6 always-on-top dual-pane window
           top pane: Japanese transcript
           bottom pane: English translation
```

Each stage runs on its own thread. The ASR and MT threads pull from bounded queues, so back-pressure is handled automatically and the UI stays responsive even if MT falls behind.

### Stage details

| Stage | Default model | Size | Notes |
|---|---|---|---|
| **Audio** | system loopback | — | WASAPI on Windows, PulseAudio/PipeWire monitor on Linux |
| **VAD** | Silero VAD (ONNX) | ~2 MB | 512-sample windows @ 16 kHz; speech threshold 0.5 |
| **ASR** | `deepdml/faster-whisper-large-v3-turbo-ct2` (int8) | ~1.5 GB | greedy decode (`beam_size=1`) for speed; swap to `kotoba-whisper-v2.0-faster` for GPU |
| **MT** | `LiquidAI/LFM2.5-1.2B-JP-GGUF` Q4_K_M | ~731 MB | JA→EN specialist; fallback: `Qwen2.5-1.5B-Instruct` Q4_K_M |
| **UI** | PySide6 overlay | — | always-on-top, dual-pane, transparent background |

### Why threads, not async?

`faster-whisper` and `llama-cpp-python` both release the GIL during their native compute. Threading gives real parallelism across the three stages without the complexity of multiprocessing or the overhead of asyncio bridging C extensions.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for environment management
- **Linux**: PulseAudio or PipeWire-pulse (default on Ubuntu 22.04+, Fedora 35+, modern Arch)
- **Windows**: nothing extra; WASAPI loopback works out of the box

## Setup

```bash
uv sync
uv run translate
```

First run downloads ~2 GB of model weights to the HuggingFace cache (`~/.cache/huggingface`). Subsequent starts load from disk in a few seconds.

## Configuration

All tuneable parameters live in [src/translate/config.py](src/translate/config.py). Key knobs:

- `ASRConfig.model` — swap to a smaller model (`faster-whisper-small`) for faster CPU inference at the cost of accuracy
- `ASRConfig.device` — set to `"cuda"` to offload Whisper to GPU and switch to `kotoba-whisper-v2.0-faster` for best JA quality
- `VADConfig.silence_ms` — lower to finalize chunks faster (more responsive, shorter context per chunk)
- `MTConfig.context_pairs` — number of previous (JA, EN) pairs fed as context into the MT model (default 4)

## Status

Core pipeline implemented and wired. Models load and run; the overlay displays live output. Further work: hotkey toggling, per-language confidence filtering, streaming partial transcripts.
