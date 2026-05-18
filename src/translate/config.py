from dataclasses import dataclass, field


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 100


@dataclass(frozen=True)
class VADConfig:
    silence_ms: int = 300
    min_chunk_ms: int = 1000
    max_chunk_ms: int = 8000
    speech_threshold: float = 0.5


@dataclass(frozen=True)
class ASRConfig:
    # CPU-viable default. kotoba-whisper (large-v3) is higher quality but
    # runs at ~0.5–1× real-time on CPU — use it only with a GPU.
    # For CPU: "Systran/faster-whisper-small" (~0.4 GB, ~6× real-time on CPU)
    #          "Systran/faster-whisper-medium" (~1.5 GB, ~3× real-time on CPU)
    model: str = "Systran/faster-whisper-small"
    compute_type: str = "int8"
    device: str = "cpu"  # cpu | cuda | auto
    language: str = "ja"
    initial_prompt: str | None = None
    beam_size: int = 1  # greedy; beam_size=5 is 5× slower on CPU with minimal quality gain
    large_model: str = "kotoba-tech/kotoba-whisper-v2.0-faster"  # JA-optimized large-v3; best with GPU
    bilingual_model: str = "kotoba-tech/kotoba-whisper-bilingual-v1.0-faster"  # JA↔EN in one model; use with --bilingual


@dataclass(frozen=True)
class MTConfig:
    model_repo: str = "LiquidAI/LFM2.5-1.2B-JP-GGUF"
    # Glob — llama-cpp-python's from_pretrained accepts a filename pattern.
    model_file: str = "*Q4_K_M.gguf"
    n_ctx: int = 4096
    n_threads: int = 0  # 0 → auto
    repeat_penalty: float = 1.05  # per LFM2.5-JP model card
    temperature: float = 0.2
    max_tokens: int = 256
    context_pairs: int = 4  # rolling (ja, en) pairs fed back as MT context
    fallback_repo: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    fallback_file: str = "*q4_k_m.gguf"


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    mt: MTConfig = field(default_factory=MTConfig)
