from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str
    duration_s: float


class ASRWorker(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> Transcript: ...
