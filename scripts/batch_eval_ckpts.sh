#!/bin/bash
set -euo pipefail

# Batch evaluate all checkpoints under /home/a25278/Workspaces/TactileWS/ckpts
# Usage: bash scripts/batch_eval_ckpts.sh [total_num]

TOTAL_NUM=${1:-20}
CKPT_BASE="/home/a25278/Workspaces/TactileWS/ckpts"
DEPLOY_YML="policy/OpenPI/deploy.yml"
LOG_DIR="batch_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES

declare -A TASK_MAP
TASK_MAP["insert_hdmi"]="insert_HDMI"
TASK_MAP["lift_can"]="lift_can"

echo "=== Batch Eval: ${TOTAL_NUM} seeds/ckpt | logs: ${LOG_DIR} ==="

for ckpt_dir in $(find "$CKPT_BASE" -name "model.safetensors" -type f 2>/dev/null | sed 's|/model\.safetensors||' | sort); do
    ckpt_name=$(basename "$ckpt_dir")
    parent=$(basename "$(dirname "$ckpt_dir")")
    label="${parent}/${ckpt_name}"

    task=""
    for pattern in "${!TASK_MAP[@]}"; do
        if [[ "$ckpt_dir" == *"$pattern"* ]]; then
            task="${TASK_MAP[$pattern]}"
            break
        fi
    done
    [ -z "$task" ] && { echo "SKIP $label — unknown task"; continue; }

    logfile="${LOG_DIR}/${task}_${parent}_${ckpt_name}.log"
    echo "[$(date '+%H:%M:%S')] Running: $label (task=$task)" | tee "$logfile"

    sed -i "s|^checkpoint_dir:.*|checkpoint_dir: ${ckpt_dir}|" "$DEPLOY_YML"

    python scripts/eval_policy.py "$task" default OpenPI/deploy --total_num "$TOTAL_NUM" \
        >> "$logfile" 2>&1

    # Show result
    grep -E "Final Result|Model loaded|\[OpenPI\] task=" "$logfile" | tail -5
    echo ""
done

echo "=== DONE | logs: ${LOG_DIR} ==="
grep -H "Final Result" ${LOG_DIR}/*.log 2>/dev/null || echo "(no results)"
