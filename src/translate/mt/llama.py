from collections import deque

from llama_cpp import Llama

from translate.config import MTConfig
from translate.mt.base import MTWorker, Translation
from translate.mt.prompt import build_messages


class LlamaMT(MTWorker):
    """llama-cpp-python backend. Loads the GGUF lazily on first translate()."""

    def __init__(self, cfg: MTConfig) -> None:
        self.cfg = cfg
        self._llm: Llama | None = None
        self._context: deque[tuple[str, str]] = deque(maxlen=cfg.context_pairs)

    def _ensure_loaded(self) -> Llama:
        if self._llm is None:
            self._llm = Llama.from_pretrained(
                repo_id=self.cfg.model_repo,
                filename=self.cfg.model_file,
                n_ctx=self.cfg.n_ctx,
                n_threads=self.cfg.n_threads or None,
                verbose=False,
            )
        return self._llm

    def prewarm(self) -> None:
        """Force model download + load now (instead of on first translate)."""
        self._ensure_loaded()

    def translate(self, text_ja: str) -> Translation:
        llm = self._ensure_loaded()
        resp = llm.create_chat_completion(
            messages=build_messages(text_ja, list(self._context)),
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            repeat_penalty=self.cfg.repeat_penalty,
        )
        en = resp["choices"][0]["message"]["content"].strip()
        # Only add to context if output looks valid (not a repetition loop).
        if en and not _is_repetition(en):
            self._context.append((text_ja, en))
        return Translation(source=text_ja, target=en)


def _is_repetition(text: str, threshold: int = 4) -> bool:
    """Detect if the model got stuck repeating the same sentence."""
    sentences = [s.strip() for s in text.replace(".", ".\n").splitlines() if s.strip()]
    if len(sentences) < threshold:
        return False
    return len(set(sentences)) <= 2
