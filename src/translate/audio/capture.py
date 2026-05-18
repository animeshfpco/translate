import sys
from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np


class AudioCapture(ABC):
    """Yields mono float32 frames at the configured sample rate, normalized to [-1, 1]."""

    def __init__(self, sample_rate: int, frame_ms: int) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = sample_rate * frame_ms // 1000

    @abstractmethod
    def __enter__(self) -> "AudioCapture": ...

    @abstractmethod
    def __exit__(self, exc_type, exc, tb) -> None: ...

    @abstractmethod
    def frames(self) -> Iterator[np.ndarray]: ...


def get_capture(sample_rate: int, frame_ms: int) -> AudioCapture:
    if sys.platform == "win32":
        from translate.audio.windows import WindowsLoopbackCapture

        return WindowsLoopbackCapture(sample_rate, frame_ms)
    if sys.platform.startswith("linux"):
        from translate.audio.linux import LinuxMonitorCapture

        return LinuxMonitorCapture(sample_rate, frame_ms)
    raise NotImplementedError(f"Unsupported platform: {sys.platform}")
