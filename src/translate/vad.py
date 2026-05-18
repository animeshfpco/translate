from collections.abc import Iterable, Iterator

import numpy as np
import torch
from silero_vad import load_silero_vad

from translate.config import VADConfig


class VADChunker:
    """Consumes streaming audio frames, yields finalized speech chunks.

    A chunk finalizes when either (a) trailing silence exceeds silence_ms after
    at least one speech window, or (b) accumulated audio exceeds max_chunk_ms.
    """

    # Silero VAD operates on 512-sample windows at 16 kHz.
    WINDOW_SAMPLES = 512

    def __init__(self, cfg: VADConfig, sample_rate: int) -> None:
        if sample_rate != 16000:
            raise ValueError(f"VADChunker requires 16 kHz audio, got {sample_rate}")
        self.cfg = cfg
        self.sample_rate = sample_rate
        self._vad = load_silero_vad(onnx=True)
        self._buf = np.zeros(0, dtype=np.float32)
        self._chunk: list[np.ndarray] = []
        self._chunk_samples = 0
        self._silence_samples = 0
        self._has_speech = False

        self._silence_limit = cfg.silence_ms * sample_rate // 1000
        self._min_chunk = cfg.min_chunk_ms * sample_rate // 1000
        self._max_chunk = cfg.max_chunk_ms * sample_rate // 1000

    def chunks(self, frames: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        for frame in frames:
            self._buf = np.concatenate(
                [self._buf, np.asarray(frame, dtype=np.float32)]
            )
            while self._buf.shape[0] >= self.WINDOW_SAMPLES:
                window = self._buf[: self.WINDOW_SAMPLES]
                self._buf = self._buf[self.WINDOW_SAMPLES :]
                yield from self._process_window(window)

    def _process_window(self, window: np.ndarray) -> Iterator[np.ndarray]:
        prob = self._vad(torch.from_numpy(window), self.sample_rate).item()

        self._chunk.append(window)
        self._chunk_samples += self.WINDOW_SAMPLES

        if prob >= self.cfg.speech_threshold:
            self._silence_samples = 0
            self._has_speech = True
        else:
            self._silence_samples += self.WINDOW_SAMPLES

        # Finalize on trailing silence after real speech.
        if (
            self._has_speech
            and self._silence_samples >= self._silence_limit
            and self._chunk_samples >= self._min_chunk
        ):
            yield np.concatenate(self._chunk)
            self._reset()
            return

        # Hard cap — emit whatever we have, drop if it was all silence.
        if self._chunk_samples >= self._max_chunk:
            if self._has_speech:
                yield np.concatenate(self._chunk)
            self._reset()

    def _reset(self) -> None:
        self._chunk = []
        self._chunk_samples = 0
        self._silence_samples = 0
        self._has_speech = False
