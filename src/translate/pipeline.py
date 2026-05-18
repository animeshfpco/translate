import logging
import threading
import time
from queue import Queue

import numpy as np

from translate.asr.base import Transcript
from translate.asr.whisper import WhisperASR
from translate.audio import get_capture
from translate.config import Config
from translate.mt.llama import LlamaMT
from translate.ui.overlay import OverlayWindow
from translate.vad import VADChunker

log = logging.getLogger("translate.pipeline")

_STOP = object()


class Pipeline:
    """Wires capture → VAD → ASR → MT → UI on three worker threads.

    Each stage releases the GIL during its native compute, so threads (not
    asyncio) get real parallelism. The Qt event loop owns the main thread.
    """

    def __init__(self, cfg: Config, bilingual: bool = False) -> None:
        self.cfg = cfg
        self.bilingual = bilingual
        self._asr_q: Queue = Queue(maxsize=8)
        self._mt_q: Queue = Queue(maxsize=8)
        self._stop = threading.Event()

    def run(self) -> None:
        self._ui = OverlayWindow()
        ui = self._ui
        asr = WhisperASR(self.cfg.asr)
        mt = None if self.bilingual else LlamaMT(self.cfg.mt)

        workers = [
            threading.Thread(target=self._prewarm, args=(asr, mt), daemon=True, name="prewarm"),
            threading.Thread(target=self._capture_loop, daemon=True, name="capture"),
            threading.Thread(target=self._asr_loop, args=(asr,), daemon=True, name="asr"),
        ]
        if mt is not None:
            workers.append(threading.Thread(target=self._mt_loop, args=(mt, ui), daemon=True, name="mt"))
        for w in workers:
            w.start()

        try:
            ui.run()
        finally:
            self._stop.set()
            self._asr_q.put(_STOP)
            self._mt_q.put(_STOP)

    def _prewarm(self, asr: WhisperASR, mt: LlamaMT | None) -> None:
        log.info("prewarm: downloading + loading ASR model (%s)", self.cfg.asr.model)
        t0 = time.monotonic()
        asr.prewarm()
        log.info("prewarm: ASR ready in %.1fs", time.monotonic() - t0)

        if mt is not None:
            log.info("prewarm: downloading + loading MT model (%s)", self.cfg.mt.model_repo)
            t0 = time.monotonic()
            mt.prewarm()
            log.info("prewarm: MT ready in %.1fs", time.monotonic() - t0)

    def _capture_loop(self) -> None:
        capture = get_capture(self.cfg.audio.sample_rate, self.cfg.audio.frame_ms)
        vad = VADChunker(self.cfg.vad, self.cfg.audio.sample_rate)
        log.info("capture: opening audio device")
        frame_count = 0
        last_log = time.monotonic()
        with capture:
            log.info("capture: device open; waiting for audio")
            for frame in capture.frames():
                if self._stop.is_set():
                    return
                frame_count += 1
                now = time.monotonic()
                if now - last_log >= 5.0:
                    rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
                    log.info("capture: %d frames in last 5s, last-frame RMS=%.4f", frame_count, rms)
                    frame_count = 0
                    last_log = now
                for chunk in vad.chunks([frame]):
                    log.info("vad: chunk finalized (%.2fs of audio)", chunk.shape[0] / self.cfg.audio.sample_rate)
                    self._asr_q.put(chunk)

    def _asr_loop(self, asr: WhisperASR) -> None:
        while not self._stop.is_set():
            item = self._asr_q.get()
            if item is _STOP:
                return
            chunk: np.ndarray = item
            try:
                t0 = time.monotonic()
                if self.bilingual:
                    transcript = asr.translate(chunk)
                    log.info("asr(translate): %.2fs audio → %.2fs compute → %r", transcript.duration_s, time.monotonic() - t0, transcript.text)
                    if transcript.text:
                        self._ui.update_last_en(transcript.text)
                else:
                    transcript: Transcript = asr.transcribe(chunk)
                    log.info("asr: %.2fs audio → %.2fs compute → %r", transcript.duration_s, time.monotonic() - t0, transcript.text)
                    if transcript.text:
                        self._ui.append_ja(transcript.text)
                        self._mt_q.put(transcript)
            except Exception:
                log.exception("asr: transcribe failed, skipping chunk")

    def _mt_loop(self, mt: LlamaMT, ui: OverlayWindow) -> None:
        while not self._stop.is_set():
            item = self._mt_q.get()
            if item is _STOP:
                return
            transcript: Transcript = item
            try:
                t0 = time.monotonic()
                tr = mt.translate(transcript.text)
                log.info("mt: %.2fs → %r", time.monotonic() - t0, tr.target)
                ui.update_last_en(tr.target)
            except Exception:
                log.exception("mt: translate failed")
                ui.update_last_en("(translation failed)")
