"""CTranslate2 NLLB backend.

Same model family as NllbMT but runs on the CTranslate2 runtime — int8/AVX2
optimized CPU kernels, typically 4–8× faster than PyTorch fp32 on CPU.

Converts the vanilla HF NLLB checkpoint to CTranslate2 format on first prewarm
and caches under ~/.cache/translate/ct2-mt/{model_repo}-{compute_type}/.
"""
from pathlib import Path

import ctranslate2
from transformers import AutoTokenizer

from translate.config import CT2MTConfig
from translate.mt.base import MTWorker, Translation


class CT2NllbMT(MTWorker):
    def __init__(self, cfg: CT2MTConfig) -> None:
        self.cfg = cfg
        self._tokenizer: AutoTokenizer | None = None
        self._translator: ctranslate2.Translator | None = None
        self._tgt_token: str = cfg.tgt_lang

    def _model_dir(self) -> Path:
        slug = self.cfg.model_repo.replace("/", "__")
        return Path.home() / ".cache" / "translate" / "ct2-mt" / f"{slug}-{self.cfg.compute_type}"

    def _ensure_converted(self) -> Path:
        target = self._model_dir()
        if (target / "model.bin").exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        from ctranslate2.converters import TransformersConverter
        converter = TransformersConverter(self.cfg.model_repo)
        converter.convert(str(target), quantization=self.cfg.compute_type, force=False)
        return target

    def _ensure_loaded(self) -> tuple[AutoTokenizer, ctranslate2.Translator]:
        if self._translator is None:
            model_dir = self._ensure_converted()
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model_repo,
                src_lang=self.cfg.src_lang,
            )
            device = _resolve_device(self.cfg.device)
            self._translator = ctranslate2.Translator(
                str(model_dir),
                device=device,
                compute_type=self.cfg.compute_type,
            )
        return self._tokenizer, self._translator

    def prewarm(self) -> None:
        self._ensure_loaded()

    def translate(self, text_ja: str) -> Translation:
        tokenizer, translator = self._ensure_loaded()
        # NLLB on CT2 expects subword tokens (strings), not ids, with the source
        # language token prefixed automatically by the tokenizer (src_lang).
        source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text_ja))
        results = translator.translate_batch(
            [source_tokens],
            target_prefix=[[self._tgt_token]],
            beam_size=self.cfg.num_beams,
            max_decoding_length=self.cfg.max_tokens,
        )
        # First target token is the language tag — drop it before decoding.
        out_tokens = results[0].hypotheses[0][1:]
        en = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(out_tokens),
            skip_special_tokens=True,
        ).strip()
        return Translation(source=text_ja, target=en)


def _resolve_device(name: str) -> str:
    if name == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return name
