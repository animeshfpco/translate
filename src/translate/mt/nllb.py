import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from translate.config import MTConfig
from translate.mt.base import MTWorker, Translation


class NllbMT(MTWorker):
    """Hugging Face NLLB-200 backend.

    Sentence-level seq2seq. No rolling context — NLLB is purpose-built for
    translation, so each utterance is translated independently.
    """

    def __init__(self, cfg: MTConfig) -> None:
        self.cfg = cfg
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForSeq2SeqLM | None = None
        self._tgt_bos_id: int | None = None
        self._device = _resolve_device(cfg.device)
        self._dtype = torch.float16 if self._device.type == "cuda" else torch.float32

    def _ensure_loaded(self) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model_repo,
                src_lang=self.cfg.src_lang,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.cfg.model_repo,
                torch_dtype=self._dtype,
            )
            self._model = model.to(self._device).eval()
            self._tgt_bos_id = self._tokenizer.convert_tokens_to_ids(self.cfg.tgt_lang)
        return self._tokenizer, self._model

    def prewarm(self) -> None:
        self._ensure_loaded()

    @torch.inference_mode()
    def translate(self, text_ja: str) -> Translation:
        tokenizer, model = self._ensure_loaded()
        inputs = tokenizer(text_ja, return_tensors="pt", truncation=True).to(self._device)
        out = model.generate(
            **inputs,
            forced_bos_token_id=self._tgt_bos_id,
            max_new_tokens=self.cfg.max_tokens,
            num_beams=self.cfg.num_beams,
            no_repeat_ngram_size=3,
        )
        en = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        return Translation(source=text_ja, target=en)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
