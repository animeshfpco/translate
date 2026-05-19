#!/usr/bin/env bash
# Run the ASR + MT pipeline on a local audio file and append results to data/results/<stem>.json.
#
# Usage:
#   ./scripts/run_file.sh <audio-file> [--device cpu|cuda|auto]
#                                      [--asr-backend ctranslate2|openvino]
#                                      [--ov-device CPU|GPU|NPU|AUTO]
#                                      [--asr-model <HF-repo-or-path>]
#                                      [--mt-model  <HF-repo>]
#                                      [--out-dir   <results-dir>]
#                                      [--bilingual [--no-transcription]]
#
# Modes:
#   Default        Whisper ASR (JA) → NLLB-200 distilled 1.3B (EN). Two models.
#   --bilingual    ASR-only translation via Whisper task=translate (use with a
#                  bilingual model). Skips the MT stage entirely.
#   --no-transcription
#                  In --bilingual mode, skip the extra JA-source transcription
#                  pass (faster; only the EN translation is emitted).
#
# ASR backends:
#   ctranslate2 (default)  CPU/CUDA via faster-whisper. Best on x86 + NVIDIA.
#   openvino               Intel NPU/iGPU/CPU via optimum-intel + OpenVINO.
#                          Best on Core Ultra laptops without dGPU.
#
# Examples:
#   ./scripts/run_file.sh data/audio/sample.wav
#   ./scripts/run_file.sh data/audio/sample.wav --device cuda
#   ./scripts/run_file.sh data/audio/sample.wav --asr-model Systran/faster-whisper-medium
#   ./scripts/run_file.sh data/audio/sample.wav --bilingual \
#       --asr-model kotoba-tech/kotoba-whisper-bilingual-v1.0-faster
#   ./scripts/run_file.sh data/audio/sample.wav --bilingual --no-transcription \
#       --asr-model kotoba-tech/kotoba-whisper-bilingual-v1.0-faster
#
# Output is a JSON array at data/results/<audio-stem>.json; each invocation
# appends one record containing model names, VAD chunks, per-chunk transcription
# and translation, timing breakdowns, and a UTC timestamp.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <audio-file> [extra args...]"
    echo ""
    echo "Examples:"
    echo "  $0 data/audio/sample.wav"
    echo "  $0 data/audio/sample.wav --device cuda"
    echo "  $0 data/audio/sample.wav --asr-model Systran/faster-whisper-medium --device cpu"
    echo ""
    echo "Extra args are passed directly to translate.run_file:"
    echo "  --asr-backend B     ctranslate2 (default) | openvino"
    echo "  --asr-model MODEL   HF repo or path for ASR"
    echo "  --mt-model  MODEL   HF repo for MT (default: facebook/nllb-200-distilled-1.3B)"
    echo "  --device    DEVICE  cpu | cuda | auto — ctranslate2 ASR + torch MT"
    echo "  --ov-device DEVICE  CPU | GPU | NPU | AUTO — OpenVINO ASR"
    echo "  --out-dir   DIR     Output directory for JSON results (default: data/results/)"
    echo "  --bilingual         Use Whisper task=translate directly (skip MT)"
    echo "  --no-transcription  In --bilingual mode, skip the JA-source pass"
    exit 1
fi

if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

python -m translate.run_file "$@"
