"""
Run the ASR + MT pipeline on a local audio file and append results to
data/results/<audio_stem>.json.

Usage:
    python -m translate.run_file path/to/audio.wav
    python -m translate.run_file path/to/audio.wav --device cuda --asr-model Systran/faster-whisper-medium
    python -m translate.run_file path/to/audio.wav --bilingual
    python -m translate.run_file path/to/audio.wav --asr-backend openvino --ov-device NPU
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from translate.asr.base import ASRWorker
from translate.asr.whisper import WhisperASR
from translate.config import ASRConfig, Config, MTConfig, OpenVINOASRConfig
from translate.mt.nllb import NllbMT
from translate.vad import VADChunker

log = logging.getLogger("translate.run_file")

RESULTS_DIR = Path(__file__).parents[2] / "data" / "results"


def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load any soundfile-supported audio file, resample to target_sr, mono."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # stereo → mono
    if sr != target_sr:
        try:
            import resampy
            audio = resampy.resample(audio, sr, target_sr)
        except ImportError:
            raise RuntimeError(
                f"Audio sample rate {sr} != {target_sr} Hz. "
                "Install resampy to auto-resample: pip install resampy"
            )
    return audio


def chunk_audio_by_frames(audio: np.ndarray, frame_samples: int) -> list[np.ndarray]:
    """Split flat audio array into fixed-size frames for VAD."""
    frames = []
    for start in range(0, len(audio), frame_samples):
        frame = audio[start : start + frame_samples]
        if len(frame) < frame_samples:
            frame = np.pad(frame, (0, frame_samples - len(frame)))
        frames.append(frame)
    return frames


def build_asr(cfg: Config, backend: str) -> ASRWorker:
    if backend == "openvino":
        from translate.asr.openvino import OpenVINOWhisperASR
        return OpenVINOWhisperASR(cfg.openvino_asr)
    return WhisperASR(cfg.asr)


def run(
    audio_path: Path,
    cfg: Config,
    bilingual: bool = False,
    asr_backend: str = "ctranslate2",
    with_transcription: bool = True,
) -> dict:
    """Run pipeline on `audio_path`.

    Modes:
      - Default (bilingual=False): ASR (JA) → NllbMT (EN). Two models.
      - Bilingual (bilingual=True): ASR-only model emits EN directly (Whisper
        task="translate"). If with_transcription=True, do a second ASR pass for
        the JA source. No MT.
    """
    asr = build_asr(cfg, asr_backend)
    mt = None if bilingual else NllbMT(cfg.mt)
    vad = VADChunker(cfg.vad, cfg.audio.sample_rate)

    log.info("Loading audio: %s", audio_path)
    audio = load_audio(audio_path, cfg.audio.sample_rate)
    total_duration_s = len(audio) / cfg.audio.sample_rate
    log.info("Loaded %.2fs of audio", total_duration_s)

    asr_label = (
        f"openvino:{cfg.openvino_asr.model} on {cfg.openvino_asr.device}"
        if asr_backend == "openvino"
        else f"ctranslate2:{cfg.asr.model} on {cfg.asr.device}"
    )
    log.info("Warming up ASR (%s)…", asr_label)
    t0 = time.monotonic()
    asr.prewarm()
    asr_load_s = time.monotonic() - t0

    mt_load_s = 0.0
    if mt is not None:
        log.info("Warming up MT model (%s)…", cfg.mt.model_repo)
        t0 = time.monotonic()
        mt.prewarm()
        mt_load_s = time.monotonic() - t0

    frame_samples = cfg.audio.sample_rate * cfg.audio.frame_ms // 1000
    frames = chunk_audio_by_frames(audio, frame_samples)
    chunks = list(vad.chunks(frames))
    log.info("VAD produced %d chunk(s)", len(chunks))

    results_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_duration_s = len(chunk) / cfg.audio.sample_rate
        log.info("Chunk %d/%d (%.2fs)…", i + 1, len(chunks), chunk_duration_s)

        transcription_text = ""
        translation_text = ""
        asr_time_s = 0.0
        mt_time_s = 0.0
        asr_duration_s = chunk_duration_s
        detected_language = cfg.asr.language

        if bilingual:
            t0 = time.monotonic()
            translation = asr.translate(chunk)
            mt_time_s = time.monotonic() - t0
            translation_text = translation.text
            detected_language = translation.language
            asr_duration_s = translation.duration_s
            log.info("  translated (ASR) in %.2fs: %r", mt_time_s, translation_text)

            if with_transcription and translation_text:
                t0 = time.monotonic()
                transcript = asr.transcribe(chunk)
                asr_time_s = time.monotonic() - t0
                transcription_text = transcript.text
                log.info("  transcribed in %.2fs: %r", asr_time_s, transcription_text)
        else:
            t0 = time.monotonic()
            transcript = asr.transcribe(chunk)
            asr_time_s = time.monotonic() - t0
            transcription_text = transcript.text
            detected_language = transcript.language
            asr_duration_s = transcript.duration_s
            log.info("  transcribed in %.2fs: %r", asr_time_s, transcription_text)

            if transcription_text:
                t0 = time.monotonic()
                tr = mt.translate(transcription_text)
                mt_time_s = time.monotonic() - t0
                translation_text = tr.target
                log.info("  translated (NLLB) in %.2fs: %r", mt_time_s, translation_text)

        results_chunks.append(
            {
                "chunk_index": i,
                "chunk_duration_s": round(chunk_duration_s, 4),
                "transcription": transcription_text,
                "detected_language": detected_language,
                "asr_duration_s": asr_duration_s,
                "asr_time_s": round(asr_time_s, 4),
                "translation": translation_text,
                "mt_time_s": round(mt_time_s, 4),
            }
        )

    return {
        "file_name": audio_path.name,
        "file_path": str(audio_path.resolve()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "bilingual" if bilingual else "default",
        "asr_backend": asr_backend,
        "with_transcription": with_transcription if bilingual else True,
        "total_audio_duration_s": round(total_duration_s, 4),
        "num_chunks": len(chunks),
        "models": {
            "asr_model": cfg.openvino_asr.model if asr_backend == "openvino" else cfg.asr.model,
            "asr_device": cfg.openvino_asr.device if asr_backend == "openvino" else cfg.asr.device,
            "asr_compute_type": None if asr_backend == "openvino" else cfg.asr.compute_type,
            "asr_language": cfg.asr.language,
            "mt_model_repo": None if bilingual else cfg.mt.model_repo,
        },
        "load_times": {
            "asr_load_s": round(asr_load_s, 4),
            "mt_load_s": round(mt_load_s, 4),
        },
        "vad_config": {
            "silence_ms": cfg.vad.silence_ms,
            "min_chunk_ms": cfg.vad.min_chunk_ms,
            "max_chunk_ms": cfg.vad.max_chunk_ms,
            "speech_threshold": cfg.vad.speech_threshold,
        },
        "chunks": results_chunks,
    }


def append_result(results_path: Path, record: dict) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if results_path.exists():
        with results_path.open() as f:
            try:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
            except json.JSONDecodeError:
                log.warning("Existing results file is malformed; starting fresh list.")
    existing.append(record)
    with results_path.open("w") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    log.info("Appended result to %s (%d total run(s))", results_path, len(existing))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run translate pipeline on a local audio file.")
    parser.add_argument("audio", type=Path, help="Path to audio file (wav/mp3/flac/…)")
    parser.add_argument(
        "--asr-backend",
        choices=["ctranslate2", "openvino"],
        default="ctranslate2",
        help="ASR runtime. 'openvino' targets Intel NPU/iGPU/CPU.",
    )
    parser.add_argument("--asr-model", default=None, help="Override ASR model (HF repo or path)")
    parser.add_argument("--mt-model", default=None, help="Override MT model repo")
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="Device for ctranslate2 ASR and torch MT.",
    )
    parser.add_argument(
        "--ov-device",
        default=None,
        help="OpenVINO device: CPU | GPU | NPU | AUTO | HETERO:NPU,CPU.",
    )
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="Directory for JSON results")
    parser.add_argument(
        "--bilingual",
        action="store_true",
        help="Use ASR model directly for translation (Whisper task=translate); skips MT.",
    )
    parser.add_argument(
        "--no-transcription",
        action="store_true",
        help="In --bilingual mode, skip the second pass that produces JA source text.",
    )
    args = parser.parse_args()

    audio_path: Path = args.audio.resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    cfg = Config()
    if args.bilingual and not args.asr_model and args.asr_backend == "ctranslate2":
        args.asr_model = cfg.asr.bilingual_model

    asr_overrides: dict[str, str] = {}
    if args.asr_model and args.asr_backend == "ctranslate2":
        asr_overrides["model"] = args.asr_model
    if args.device:
        asr_overrides["device"] = args.device

    mt_overrides: dict[str, str] = {}
    if args.mt_model:
        mt_overrides["model_repo"] = args.mt_model
    if args.device:
        mt_overrides["device"] = args.device

    ov_overrides: dict[str, str] = {}
    if args.asr_model and args.asr_backend == "openvino":
        ov_overrides["model"] = args.asr_model
        if args.bilingual:
            ov_overrides["bilingual_model"] = args.asr_model
    if args.ov_device:
        ov_overrides["device"] = args.ov_device
    if args.bilingual and args.asr_backend == "openvino" and "model" not in ov_overrides:
        ov_overrides["model"] = cfg.openvino_asr.bilingual_model

    if asr_overrides or mt_overrides or ov_overrides:
        cfg = Config(
            audio=cfg.audio,
            vad=cfg.vad,
            asr=ASRConfig(**{**cfg.asr.__dict__, **asr_overrides}),
            openvino_asr=OpenVINOASRConfig(**{**cfg.openvino_asr.__dict__, **ov_overrides}),
            mt=MTConfig(**{**cfg.mt.__dict__, **mt_overrides}),
        )

    record = run(
        audio_path,
        cfg,
        bilingual=args.bilingual,
        asr_backend=args.asr_backend,
        with_transcription=not args.no_transcription,
    )

    results_path = args.out_dir / f"{audio_path.stem}.json"
    append_result(results_path, record)
    print(f"Results written to: {results_path}")


if __name__ == "__main__":
    main()
