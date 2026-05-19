import argparse
import logging

from translate.config import ASRConfig, CT2MTConfig, Config, MTConfig, OpenVINOASRConfig
from translate.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time JA→EN translator overlay.")
    parser.add_argument(
        "--bilingual",
        action="store_true",
        help="Use Whisper task=translate directly; skips MT.",
    )
    parser.add_argument(
        "--asr-backend",
        choices=["ctranslate2", "openvino"],
        default="ctranslate2",
        help="ASR runtime. 'openvino' targets Intel NPU/iGPU/CPU; requires "
             "the 'openvino' extra (uv sync --extra openvino).",
    )
    parser.add_argument(
        "--mt-backend",
        choices=["torch", "ctranslate2"],
        default="ctranslate2",
        help="MT runtime. 'ctranslate2' is int8 + AVX2-optimized — typically 4–8× "
             "faster than 'torch' on CPU. Auto-converts NLLB to CT2 on first run.",
    )
    parser.add_argument(
        "--mt-model",
        default=None,
        help="Override MT model (HF repo). Applied to whichever --mt-backend is active.",
    )
    parser.add_argument(
        "--mt-compute-type",
        default=None,
        help="CT2 quantization: int8 | int8_float16 | float16 | float32 (when --mt-backend ctranslate2).",
    )
    parser.add_argument("--asr-model", default=None, help="Override ASR model (HF repo or path)")
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="Device for ctranslate2 ASR and torch MT (NLLB).",
    )
    parser.add_argument(
        "--ov-device",
        default=None,
        help="OpenVINO device: CPU | GPU | NPU | AUTO | HETERO:NPU,CPU (when --asr-backend openvino).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = Config()

    # ctranslate2 model override: in bilingual mode default to the bilingual variant.
    if args.bilingual and not args.asr_model and args.asr_backend == "ctranslate2":
        args.asr_model = cfg.asr.bilingual_model

    asr_overrides: dict[str, str] = {}
    if args.asr_model and args.asr_backend == "ctranslate2":
        asr_overrides["model"] = args.asr_model
    if args.device:
        asr_overrides["device"] = args.device

    mt_overrides: dict[str, str] = {}
    if args.device:
        mt_overrides["device"] = args.device
    if args.mt_model and args.mt_backend == "torch":
        mt_overrides["model_repo"] = args.mt_model

    ct2_mt_overrides: dict[str, str] = {}
    if args.device:
        ct2_mt_overrides["device"] = args.device
    if args.mt_model and args.mt_backend == "ctranslate2":
        ct2_mt_overrides["model_repo"] = args.mt_model
    if args.mt_compute_type:
        ct2_mt_overrides["compute_type"] = args.mt_compute_type

    ov_overrides: dict[str, str] = {}
    if args.asr_model and args.asr_backend == "openvino":
        ov_overrides["model"] = args.asr_model
        if args.bilingual:
            ov_overrides["bilingual_model"] = args.asr_model
    if args.ov_device:
        ov_overrides["device"] = args.ov_device
    # In openvino + bilingual mode default the model to the bilingual variant.
    if args.bilingual and args.asr_backend == "openvino" and "model" not in ov_overrides:
        ov_overrides["model"] = cfg.openvino_asr.bilingual_model

    if asr_overrides or mt_overrides or ct2_mt_overrides or ov_overrides:
        cfg = Config(
            audio=cfg.audio,
            vad=cfg.vad,
            asr=ASRConfig(**{**cfg.asr.__dict__, **asr_overrides}),
            openvino_asr=OpenVINOASRConfig(**{**cfg.openvino_asr.__dict__, **ov_overrides}),
            mt=MTConfig(**{**cfg.mt.__dict__, **mt_overrides}),
            ct2_mt=CT2MTConfig(**{**cfg.ct2_mt.__dict__, **ct2_mt_overrides}),
        )

    mode = "bilingual" if args.bilingual else "default"
    Pipeline(cfg, mode=mode, asr_backend=args.asr_backend, mt_backend=args.mt_backend).run()


if __name__ == "__main__":
    main()
