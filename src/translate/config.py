from dataclasses import dataclass, field


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 100


@dataclass(frozen=True)
class VADConfig:
    silence_ms: int = 850
    min_chunk_ms: int = 1000
    max_chunk_ms: int = 20000
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
    bilingual_model: str = "Systran/faster-whisper-small"  # JA↔EN in one model; use with --bilingual


@dataclass(frozen=True)
class MTConfig:
    # 600M is the CPU-real-time default. 1.3B is higher quality but ~2× slower —
    # use it only if MT can keep up with ASR on the target machine.
    model_repo: str = "facebook/nllb-200-distilled-600M"
    src_lang: str = "jpn_Jpan"  # NLLB BCP-47 + script tag
    tgt_lang: str = "eng_Latn"
    device: str = "cpu"  # cpu | cuda | auto
    num_beams: int = 1  # greedy; beam=4 is ~4× slower for marginal quality gain
    max_tokens: int = 256


@dataclass(frozen=True)
class CT2MTConfig:
    # Same HF repo as MTConfig — CT2 converts from the vanilla NLLB checkpoint.
    # Auto-converted on first prewarm and cached under ~/.cache/translate/ct2-mt/.
    model_repo: str = "facebook/nllb-200-distilled-600M"
    src_lang: str = "jpn_Jpan"
    tgt_lang: str = "eng_Latn"
    device: str = "cpu"  # cpu | cuda | auto
    compute_type: str = "int8"  # int8 | int8_float16 | float16 | float32
    num_beams: int = 1
    max_tokens: int = 256


@dataclass(frozen=True)
class OpenVINOASRConfig:
    # Prequantized OpenVINO IR weights from the OpenVINO org. Use the int8 small
    # variant by default; bump to int4 large-v3 if you want quality at the cost
    # of NPU memory.
    model: str = "OpenVINO/whisper-small-int8-ov"
    bilingual_model: str = "OpenVINO/whisper-small-int8-ov"
    # OpenVINO device strings: CPU | GPU | NPU | AUTO | HETERO:NPU,CPU
    # AUTO picks the best available; force NPU on Core Ultra laptops.
    device: str = "AUTO"
    language: str = "ja"
    max_new_tokens: int = 256


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    openvino_asr: OpenVINOASRConfig = field(default_factory=OpenVINOASRConfig)
    mt: MTConfig = field(default_factory=MTConfig)
    ct2_mt: CT2MTConfig = field(default_factory=CT2MTConfig)
