"""Download ASR and MT models to the HF cache with visible progress.

Run from the notebook:
    %run notebooks/download_models.py

Or from the terminal:
    uv run python notebooks/download_models.py
"""
import sys
from pathlib import Path

# Make sure the src package is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tqdm.auto import tqdm  # noqa: F401 — importing early forces notebook-aware tqdm

from translate.config import ASRConfig, MTConfig

asr_cfg = ASRConfig()
mt_cfg = MTConfig()

# ── ASR ──────────────────────────────────────────────────────────────────────
print(f"\n[1/2] ASR — {asr_cfg.model}  (compute_type={asr_cfg.compute_type})")
print("      Downloading and loading into memory...")

from faster_whisper import WhisperModel

asr_model = WhisperModel(
    asr_cfg.model,
    device=asr_cfg.device,
    compute_type=asr_cfg.compute_type,
)
print("      ✓ ASR ready")

# ── MT ───────────────────────────────────────────────────────────────────────
print(f"\n[2/2] MT  — {mt_cfg.model_repo}")
print("      Downloading and loading into memory...")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

mt_tokenizer = AutoTokenizer.from_pretrained(mt_cfg.model_repo, src_lang=mt_cfg.src_lang)
mt_model = AutoModelForSeq2SeqLM.from_pretrained(mt_cfg.model_repo)
print("      ✓ MT ready")

print("\nBoth models cached. Next run of `uv run translate` will skip the download.")
