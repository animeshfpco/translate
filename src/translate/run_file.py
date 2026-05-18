"""
Run the ASR + MT pipeline on a local audio file and append results to
data/results/<audio_stem>.json.

Usage:
    python -m translate.run_file path/to/audio.wav
    python -m translate.run_file path/to/audio.wav --device cuda --asr-model Systran/faster-whisper-medium
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from translate.asr.whisper import WhisperASR
from translate.config import ASRConfig, Config, MTConfig, VADConfig
from translate.mt.llama import LlamaMT
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


def run(audio_path: Path, cfg: Config) -> dict:
    asr = WhisperASR(cfg.asr)
    mt = LlamaMT(cfg.mt)
    vad = VADChunker(cfg.vad, cfg.audio.sample_rate)

    log.info("Loading audio: %s", audio_path)
    audio = load_audio(audio_path, cfg.audio.sample_rate)
    total_duration_s = len(audio) / cfg.audio.sample_rate
    log.info("Loaded %.2fs of audio", total_duration_s)

    log.info("Warming up ASR model (%s)…", cfg.asr.model)
    t0 = time.monotonic()
    asr.prewarm()
    asr_load_s = time.monotonic() - t0

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
        log.info("ASR chunk %d/%d (%.2fs)…", i + 1, len(chunks), chunk_duration_s)

        t0 = time.monotonic()
        transcript = asr.transcribe(chunk)
        asr_time_s = time.monotonic() - t0
        log.info("  transcribed in %.2fs: %r", asr_time_s, transcript.text)

        translation_text = ""
        mt_time_s = 0.0
        if transcript.text:
            t0 = time.monotonic()
            translation = mt.translate(transcript.text)
            mt_time_s = time.monotonic() - t0
            translation_text = translation.target
            log.info("  translated in %.2fs: %r", mt_time_s, translation_text)

        results_chunks.append(
            {
                "chunk_index": i,
                "chunk_duration_s": round(chunk_duration_s, 4),
                "transcription": transcript.text,
                "detected_language": transcript.language,
                "asr_duration_s": transcript.duration_s,
                "asr_time_s": round(asr_time_s, 4),
                "translation": translation_text,
                "mt_time_s": round(mt_time_s, 4),
            }
        )

    return {
        "file_name": audio_path.name,
        "file_path": str(audio_path.resolve()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_audio_duration_s": round(total_duration_s, 4),
        "num_chunks": len(chunks),
        "models": {
            "asr_model": cfg.asr.model,
            "asr_device": cfg.asr.device,
            "asr_compute_type": cfg.asr.compute_type,
            "asr_language": cfg.asr.language,
            "mt_model_repo": cfg.mt.model_repo,
            "mt_model_file": cfg.mt.model_file,
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
    parser.add_argument("--asr-model", default=None, help="Override ASR model (HF repo or path)")
    parser.add_argument("--mt-model", default=None, help="Override MT model repo")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"], help="ASR device")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="Directory for JSON results")
    args = parser.parse_args()

    audio_path: Path = args.audio.resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    cfg = Config()
    # Only rebuild sub-configs when CLI args explicitly override them.
    if args.asr_model or args.device:
        cfg = Config(
            audio=cfg.audio,
            vad=cfg.vad,
            asr=ASRConfig(
                **{**cfg.asr.__dict__,
                   **({"model": args.asr_model} if args.asr_model else {}),
                   **({"device": args.device} if args.device else {})}
            ),
            mt=cfg.mt,
        )
    if args.mt_model:
        cfg = Config(
            audio=cfg.audio,
            vad=cfg.vad,
            asr=cfg.asr,
            mt=MTConfig(**{**cfg.mt.__dict__, "model_repo": args.mt_model}),
        )

    record = run(audio_path, cfg)

    results_path = args.out_dir / f"{audio_path.stem}.json"
    append_result(results_path, record)
    print(f"Results written to: {results_path}")


if __name__ == "__main__":
    main()
