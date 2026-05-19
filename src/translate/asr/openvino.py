"""OpenVINO Whisper backend.

Targets Intel NPU / iGPU / CPU via the OpenVINO runtime. Uses prequantized
int8 / int4 IR weights from the OpenVINO org on the Hugging Face Hub, so
there's no on-the-fly conversion at startup.

Requires the 'openvino' extra: `uv sync --extra openvino`.
"""
import numpy as np

from translate.asr.base import ASRWorker, Transcript
from translate.config import OpenVINOASRConfig

_SR = 16000


class OpenVINOWhisperASR(ASRWorker):
    def __init__(self, cfg: OpenVINOASRConfig) -> None:
        self.cfg = cfg
        self._processor = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            # Deferred so the base install (without the openvino extra) still imports.
            from optimum.intel.openvino import OVModelForSpeechSeq2Seq
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.cfg.model)
            self._model = OVModelForSpeechSeq2Seq.from_pretrained(
                self.cfg.model,
                device=self.cfg.device,
            )
            # Force kernel compilation now so the first transcribe() doesn't pay for it.
            self._model.compile()
        return self._processor, self._model

    def prewarm(self) -> None:
        self._ensure_loaded()

    def transcribe(self, audio: np.ndarray) -> Transcript:
        return self._run(audio, task="transcribe")

    def translate(self, audio: np.ndarray) -> Transcript:
        return self._run(audio, task="translate")

    def _run(self, audio: np.ndarray, task: str) -> Transcript:
        processor, model = self._ensure_loaded()
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        duration_s = audio.shape[0] / _SR
        inputs = processor(audio, sampling_rate=_SR, return_tensors="pt")
        generated = model.generate(
            inputs["input_features"],
            language=self.cfg.language,
            task=task,
            max_new_tokens=self.cfg.max_new_tokens,
        )
        text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return Transcript(text=text, language=self.cfg.language, duration_s=duration_s)
