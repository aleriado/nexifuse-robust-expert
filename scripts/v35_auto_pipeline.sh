#!/bin/bash
# V3.5 Autonomous Pipeline Monitor
# Monitors Day 1-2 (already running), then chains Day 3-4, 5-6, 7, processing, and training.
# Run: nohup bash scripts/v35_auto_pipeline.sh > /tmp/v35_logs/auto_pipeline.log 2>&1 &

set -e
cd /home/naritadaiki3/nexifuse_project
source nexifuse_env/bin/activate

LOG_DIR="/tmp/v35_logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/auto_pipeline.log"; }

wait_for_file() {
    # Wait for a file to reach a target line count
    local file=$1
    local target=$2
    local check_interval=${3:-300}  # check every 5 min
    log "Waiting for $file to reach $target lines..."
    while true; do
        local count=0
        [ -f "$file" ] && count=$(wc -l < "$file")
        log "  $file: $count / $target"
        if [ "$count" -ge "$target" ]; then
            log "  Target reached!"
            return 0
        fi
        sleep $check_interval
    done
}

wait_for_process() {
    # Wait for a process to finish
    local pid=$1
    log "Waiting for PID $pid to finish..."
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    log "  PID $pid finished"
}

# ── Monitor Day 1-2 (already running) ────────────────────────────────
log "=== V3.5 Autonomous Pipeline Started ==="
log "=== Monitoring Day 1-2: PHI-Safe Code + Identity Edge Cases ==="

# Wait for PHI-safe generation to reach at least 5500 (allowing some failures)
wait_for_file "data/raw/v35_phi_safe.jsonl" 5500 300

# Day 1-2 PHI-safe done. Check if identity generation started (it runs after PHI in the script)
# Wait for identity edge cases
wait_for_file "data/raw/v35_identity_edge.jsonl" 1800 120

log "Day 1-2 COMPLETE: $(wc -l < data/raw/v35_phi_safe.jsonl) PHI + $(wc -l < data/raw/v35_identity_edge.jsonl) identity"

# ── Day 3-4: Bidirectional Translation + Mirth Channels ─────────────
log "=== Starting Day 3-4: Bidirectional Translation + Mirth Channels ==="
python scripts/generate_v35_day3_4.py \
    --translation-output data/raw/v35_translation.jsonl \
    --mirth-output data/raw/v35_mirth_channels.jsonl \
    --translation-count 5000 --mirth-count 3000 \
    2>&1 | tee "$LOG_DIR/day3_4.log"
log "Day 3-4 COMPLETE: $(wc -l < data/raw/v35_translation.jsonl) translation + $(wc -l < data/raw/v35_mirth_channels.jsonl) mirth"

# ── Day 5-6: Error Handling + Debug ──────────────────────────────────
log "=== Starting Day 5-6: Error Handling + Debug ==="
python scripts/generate_v35_day5_6.py \
    --error-output data/raw/v35_error_handling.jsonl \
    --debug-output data/raw/v35_debug.jsonl \
    --retrofit-output data/raw/v35_error_retrofit.jsonl \
    --error-count 5000 --debug-count 3000 --retrofit-count 10000 \
    2>&1 | tee "$LOG_DIR/day5_6.log"
log "Day 5-6 COMPLETE: $(wc -l < data/raw/v35_error_handling.jsonl) error + $(wc -l < data/raw/v35_debug.jsonl) debug + $(wc -l < data/raw/v35_error_retrofit.jsonl) retrofit"

# ── Day 7: Math + Clarification + Architecture + Vendor EHR ─────────
log "=== Starting Day 7: Math + Clarification + Architecture + Vendor EHR ==="
python scripts/generate_v35_day7.py \
    --math-output data/raw/v35_math.jsonl \
    --clarification-output data/raw/v35_clarification.jsonl \
    --architecture-output data/raw/v35_architecture.jsonl \
    --vendor-output data/raw/v35_vendor_ehr.jsonl \
    --math-count 3000 --clarification-count 2000 --architecture-count 2000 --vendor-count 3000 \
    2>&1 | tee "$LOG_DIR/day7.log"
log "Day 7 COMPLETE"

# ── Summary ──────────────────────────────────────────────────────────
log "=== All data generation complete ==="
log "File summary:"
for f in data/raw/v35_*.jsonl; do
    log "  $(basename $f): $(wc -l < $f) examples"
done
total_new=$(cat data/raw/v35_*.jsonl | wc -l)
total_existing=$(cat data/raw/synthetic.jsonl data/raw/synthetic_run1.jsonl data/raw/general.jsonl data/raw/conversations.jsonl data/raw/conceptual.jsonl data/raw/raw_hl7.jsonl data/raw/scraped.jsonl 2>/dev/null | wc -l)
log "Total new V3.5: $total_new"
log "Total existing V3: $total_existing"
log "Combined total: $((total_new + total_existing))"

# ── Data Processing ──────────────────────────────────────────────────
log "=== Starting data processing (clean + validate + format) ==="

# Unload teacher model to free GPUs for processing
curl -s http://localhost:11434/api/generate -d '{"model":"llama3:70b","keep_alive":"0"}' > /dev/null 2>&1
curl -s http://localhost:11434/api/generate -d '{"model":"llama3:8b","keep_alive":"0"}' > /dev/null 2>&1
sleep 5

python -m nexifuse clean 2>&1 | tee "$LOG_DIR/clean.log"
log "Clean complete: $(wc -l < data/cleaned/cleaned.jsonl) examples"

python -m nexifuse validate 2>&1 | tee "$LOG_DIR/validate.log"
log "Validate complete: $(wc -l < data/validated/passed.jsonl) passed, $(wc -l < data/validated/failed.jsonl) failed"

python -m nexifuse format 2>&1 | tee "$LOG_DIR/format.log"
log "Format complete: $(wc -l < data/formatted/train.jsonl) formatted examples"

# ── Round 1 SFT Training ─────────────────────────────────────────────
log "=== Starting Round 1 SFT Training (8x L4 DDP) ==="

# Free all GPU memory
curl -s http://localhost:11434/api/generate -d '{"model":"nexifuse-robust-expert","keep_alive":"0"}' > /dev/null 2>&1
sleep 5

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun \
    --nproc_per_node=8 --master_port=29505 \
    /tmp/train_wrapper.py 2>&1 | tee "$LOG_DIR/round1_training.log"

log "Round 1 SFT training complete"

# ── Export Round 1 ────────────────────────────────────────────────────
log "=== Exporting Round 1 model ==="
python -m nexifuse merge 2>&1 | tee "$LOG_DIR/round1_merge.log"
python /home/naritadaiki3/nexifuse_project/llama.cpp/convert_hf_to_gguf.py outputs/merged_model --outfile outputs/nexifuse-v35-r1-f16.gguf --outtype f16 2>&1
/home/naritadaiki3/nexifuse_project/llama.cpp/llama-quantize outputs/nexifuse-v35-r1-f16.gguf outputs/nexifuse-v35-r1-q4km.gguf Q4_K_M 2>&1
log "Round 1 export complete"

log "=== V3.5 Phase A pipeline complete up to Round 1 ==="
log "=== Next: Run mid-point benchmark, then Round 2 + ORPO ==="
log "=== Monitor progress: tail -f $LOG_DIR/auto_pipeline.log ==="
