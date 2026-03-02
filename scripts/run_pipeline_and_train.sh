#!/usr/bin/env bash
# Run full data pipeline then train.
# Usage:
#   ./scripts/run_pipeline_and_train.sh [num_per_domain]
# Example:
#   ./scripts/run_pipeline_and_train.sh 2000
#
# Prerequisites:
#   - Ollama running with teacher model (e.g. ollama run llama3:70b)
#   - config.yaml data_factory.model_name matches your Ollama model

set -e
cd "$(dirname "$0")/.."
NUM_PER_DOMAIN="${1:-500}"

echo "=== Pipeline (num-per-domain=$NUM_PER_DOMAIN) ==="
python3 -m nexifuse pipeline --num-per-domain "$NUM_PER_DOMAIN"

echo ""
echo "=== Training ==="
python3 -m nexifuse train

echo ""
echo "=== Done. Optional: convert, modelfile, register ==="
echo "  python3 -m nexifuse convert"
echo "  python3 -m nexifuse modelfile"
echo "  python3 -m nexifuse register --name nexifuse-robust-expert"
