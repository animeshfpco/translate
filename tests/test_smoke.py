def test_package_imports() -> None:
    import translate
    from translate import audio, asr, mt, ui  # noqa: F401
    from translate.config import Config

    cfg = Config()
    assert cfg.mt.model_repo == "LiquidAI/LFM2.5-1.2B-JP-GGUF"
    assert cfg.mt.repeat_penalty == 1.05
    assert cfg.asr.language == "ja"
    assert translate.__version__
