#!/usr/bin/env bash
# Generate 20k+ cleaned training examples and format with conversational data.
# Runs: ingest → scrape → generate (6000/domain) → clean → validate → format.
# Output: data/cleaned/cleaned.jsonl (~20k+ rows), data/formatted/train.jsonl (identity + passed).
#
# Prerequisites:
#   - Ollama serving teacher model (e.g. ollama run llama3:70b)
#   - config.yaml data_factory.model_name matches your Ollama model
#
# Usage: ./scripts/generate_20k_cleaned.sh

set -e
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
if [ -d "nexifuse_env" ] && [ -x "nexifuse_env/bin/python" ]; then
  PYTHON="./nexifuse_env/bin/python"
fi

echo "=== Pipeline for 20k+ cleaned (num_per_domain=6000, includes conversational) ==="
"$PYTHON" -m nexifuse pipeline-20k

echo ""
echo "=== Done. Check counts: ==="
wc -l data/cleaned/cleaned.jsonl data/validated/passed.jsonl data/formatted/train.jsonl 2>/dev/null || true
echo ""
echo "Next: python -m nexifuse train   (or train-multigpu)"
