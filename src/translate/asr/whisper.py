import numpy as np
from faster_whisper import WhisperModel

from translate.asr.base import ASRWorker, Transcript
from translate.config import ASRConfig


class WhisperASR(ASRWorker):
    """faster-whisper backend. Loads the model lazily on first transcribe()."""

    def __init__(self, cfg: ASRConfig) -> None:
        self.cfg = cfg
        self._model: WhisperModel | None = None

    def _ensure_loaded(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self.cfg.model,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
            )
        return self._model

    def prewarm(self) -> None:
        """Force model download + load now (instead of on first transcribe)."""
        self._ensure_loaded()

    def transcribe(self, audio: np.ndarray) -> Transcript:
        model = self._ensure_loaded()
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        segments, info = model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            initial_prompt=self.cfg.initial_prompt,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = "".join(s.text for s in segments).strip()
        return Transcript(text=text, language=info.language, duration_s=info.duration)
