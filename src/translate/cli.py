import argparse
import logging

from translate.config import ASRConfig, Config
from translate.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time JA→EN translator overlay.")
    parser.add_argument(
        "--bilingual",
        action="store_true",
        help="Use Whisper task=translate directly; skips LLM MT.",
    )
    parser.add_argument("--asr-model", default=None, help="Override ASR model (HF repo or path)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"], help="ASR device")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = Config()
    if args.bilingual and not args.asr_model:
        args.asr_model = cfg.asr.bilingual_model
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

    Pipeline(cfg, bilingual=args.bilingual).run()


if __name__ == "__main__":
    main()
