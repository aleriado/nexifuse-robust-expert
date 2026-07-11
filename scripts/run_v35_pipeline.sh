#!/bin/bash
# V3.5 Full Pipeline Orchestrator
# Runs all data generation days sequentially, then processes and trains.
# Usage: bash scripts/run_v35_pipeline.sh

set -e
cd /home/naritadaiki3/nexifuse_project
source nexifuse_env/bin/activate

LOG_DIR="/tmp/v35_logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/pipeline.log"; }

# ── Day 1-2: PHI-Safe Code + Identity Edge Cases ────────────────────
log "=== Day 1-2: PHI-Safe Code + Identity Edge Cases ==="
python scripts/generate_v35_phi_safe.py \
    --phi-output data/raw/v35_phi_safe.jsonl \
    --identity-output data/raw/v35_identity_edge.jsonl \
    --phi-count 6000 --identity-count 2000 \
    2>&1 | tee "$LOG_DIR/day1_2.log"
log "Day 1-2 complete: $(wc -l < data/raw/v35_phi_safe.jsonl) PHI + $(wc -l < data/raw/v35_identity_edge.jsonl) identity"

# ── Day 3-4: Bidirectional Translation + Mirth Channels ─────────────
log "=== Day 3-4: Bidirectional Translation + Mirth Channels ==="
python scripts/generate_v35_day3_4.py \
    --translation-output data/raw/v35_translation.jsonl \
    --mirth-output data/raw/v35_mirth_channels.jsonl \
    --translation-count 5000 --mirth-count 3000 \
    2>&1 | tee "$LOG_DIR/day3_4.log"
log "Day 3-4 complete: $(wc -l < data/raw/v35_translation.jsonl) translation + $(wc -l < data/raw/v35_mirth_channels.jsonl) mirth"

# ── Day 5-6: Error Handling + Debug ──────────────────────────────────
log "=== Day 5-6: Error Handling + Debug ==="
python scripts/generate_v35_day5_6.py \
    --error-output data/raw/v35_error_handling.jsonl \
    --debug-output data/raw/v35_debug.jsonl \
    --retrofit-output data/raw/v35_error_retrofit.jsonl \
    --error-count 5000 --debug-count 3000 --retrofit-count 10000 \
    2>&1 | tee "$LOG_DIR/day5_6.log"
log "Day 5-6 complete: $(wc -l < data/raw/v35_error_handling.jsonl) error + $(wc -l < data/raw/v35_debug.jsonl) debug + $(wc -l < data/raw/v35_error_retrofit.jsonl) retrofit"

# ── Day 7: Math + Clarification + Architecture + Vendor EHR ─────────
log "=== Day 7: Math + Clarification + Architecture + Vendor EHR ==="
python scripts/generate_v35_day7.py \
    --math-output data/raw/v35_math.jsonl \
    --clarification-output data/raw/v35_clarification.jsonl \
    --architecture-output data/raw/v35_architecture.jsonl \
    --vendor-output data/raw/v35_vendor_ehr.jsonl \
    --math-count 3000 --clarification-count 2000 --architecture-count 2000 --vendor-count 3000 \
    2>&1 | tee "$LOG_DIR/day7.log"
log "Day 7 complete"

# ── Day 8-9: Vendor-Specific EHR (continued if needed) ──────────────
log "=== Data generation complete. Summary: ==="
for f in data/raw/v35_*.jsonl; do
    echo "  $(basename $f): $(wc -l < $f) examples"
done | tee -a "$LOG_DIR/pipeline.log"

# ── Day 10: Data Processing ──────────────────────────────────────────
log "=== Day 10: Data Processing (clean + validate + format) ==="
python -m nexifuse clean 2>&1 | tee "$LOG_DIR/clean.log"
python -m nexifuse validate 2>&1 | tee "$LOG_DIR/validate.log"
python -m nexifuse format 2>&1 | tee "$LOG_DIR/format.log"
log "Processing complete: $(wc -l < data/formatted/train.jsonl) formatted examples"

log "=== Pipeline complete. Ready for training. ==="
echo ""
echo "Next steps:"
echo "  1. Review data/formatted/train.jsonl"
echo "  2. Run: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29505 /tmp/train_wrapper.py"
echo "  3. Run mid-point benchmark"
echo "  4. Generate ORPO pairs"
echo "  5. Run ORPO alignment"
