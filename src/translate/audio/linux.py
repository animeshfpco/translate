from collections.abc import Iterator

import numpy as np
import soundcard as sc

from translate.audio.capture import AudioCapture


class LinuxMonitorCapture(AudioCapture):
    """PulseAudio/PipeWire monitor source of the default sink, via soundcard.

    On Linux, soundcard exposes the monitor of the default speaker as a
    loopback-flagged microphone — same API as the Windows side.
    """

    def __enter__(self) -> "LinuxMonitorCapture":
        speaker = sc.default_speaker()
        mic = sc.get_microphone(speaker.name, include_loopback=True)
        self._cm = mic.recorder(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.frame_samples,
        )
        self._recorder = self._cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._cm.__exit__(exc_type, exc, tb)

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            data = self._recorder.record(numframes=self.frame_samples)
            if data.ndim > 1:
                data = data.mean(axis=1)
            yield data.astype(np.float32, copy=False)
