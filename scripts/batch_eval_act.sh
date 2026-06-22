#!/bin/bash
# Batch evaluate ACT checkpoints (vision_only variant).
#
# ACT checkpoints: /home/a25278/Workspaces/TactileWS/ckpts/act/{task}/vision_only/
# Each has: policy_last.ckpt, dataset_stats.pkl
#
# NOTE: univtac variant is for ViTAL, NOT ACT. Only test vision_only here.
#
# Usage:
#   bash scripts/batch_eval_act.sh                          # all tasks, 50 seeds
#   bash scripts/batch_eval_act.sh --max_seed 9             # 10 seeds each
#   bash scripts/batch_eval_act.sh --tasks insert_HDMI      # single task

set -euo pipefail

CKPT_ROOT="/home/a25278/Workspaces/TactileWS/ckpts/act"
PROJ_ROOT="/home/a25278/Workspaces/TactileWS/UniVTAC"
ACT_CKPT_BASE="$PROJ_ROOT/policy/ACT/act_ckpt"
RESULT_ROOT="$PROJ_ROOT/eval_result/ACT_batch"
MAX_SEED=49
START_SEED=0
TASK_FILTER=""
TRAIN_CFG="train_config_vision"   # ACT = vision_only

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max_seed)   MAX_SEED="$2";   shift 2 ;;
        --start_seed) START_SEED="$2"; shift 2 ;;
        --tasks)      TASK_FILTER="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

cd "$PROJ_ROOT"
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export TRAIN_CONFIG="$TRAIN_CFG"
mkdir -p "$RESULT_ROOT"

# ── Collect vision_only checkpoints ──────────────────────────
declare -a TASKS=()
for task_dir in "$CKPT_ROOT"/*/; do
    task=$(basename "$task_dir")
    [[ "$task" == "encoder.pth" ]] && continue
    [[ -n "$TASK_FILTER" && "$task" != "$TASK_FILTER" ]] && continue
    ckpt_dir="$task_dir/vision_only"
    [[ -f "$ckpt_dir/policy_last.ckpt" ]] || continue
    TASKS+=("$task|$ckpt_dir")
done

echo "============================================"
echo " ACT Batch Eval (vision_only)"
echo " ${#TASKS[@]} checkpoints, seeds $START_SEED → $MAX_SEED"
echo " Train config: $TRAIN_CFG"
echo " Logs: $RESULT_ROOT"
echo "============================================"

OK=0
FAIL=0

for entry in "${TASKS[@]}"; do
    IFS='|' read -r task ckpt_dir <<< "$entry"

    echo ""
    echo "===== $task ====="

    # ── Symlink checkpoint ──
    target="$ACT_CKPT_BASE/act-${task}/demo-50/${TRAIN_CFG}"
    mkdir -p "$target"
    rm -f "$target/policy_best.ckpt" "$target/dataset_stats.pkl"
    ln -sf "$ckpt_dir/policy_last.ckpt"  "$target/policy_best.ckpt"
    ln -sf "$ckpt_dir/dataset_stats.pkl" "$target/dataset_stats.pkl"

    # ── Run eval ──
    outfile="/tmp/eval_batch_${task}.txt"
    echo "  Starting..."
    t0=$(date +%s)

    if timeout 7200 python scripts/eval_policy.py "$task" demo ACT/deploy \
        --start_seed "$START_SEED" --max_seed "$MAX_SEED" \
        > "$outfile" 2>&1; then

        t1=$(date +%s)
        dt=$((t1 - t0))

        # Parse result from the log file (more reliable than stdout)
        logsrc=$(ls -td "$PROJ_ROOT/eval_result/ACT/$task/deploy/"*/log.log 2>/dev/null | head -1)
        if [[ -n "$logsrc" ]]; then
            last_line=$(grep "Total.*success" "$logsrc" | tail -1 || echo "?")
        else
            last_line="(no log)"
        fi

        echo "  Done in ${dt}s — $last_line"
        OK=$((OK + 1))

        # Save
        savedir="$RESULT_ROOT/${task}"
        mkdir -p "$savedir"
        cp "$outfile" "$savedir/output.txt"
        [[ -n "$logsrc" ]] && cp "$logsrc" "$savedir/log.log"
    else
        t1=$(date +%s)
        dt=$((t1 - t0))
        echo "  FAILED after ${dt}s"
        grep -E "Traceback|Error:" "$outfile" | head -5
        cp "$outfile" "$RESULT_ROOT/${task}_fail.txt" 2>/dev/null || true
        FAIL=$((FAIL + 1))
    fi

    # ── Cleanup GPU ──
    pkill -9 -f "python.*eval_policy" 2>/dev/null || true
    sleep 5
done

echo ""
echo "============================================"
echo " DONE — $OK ok, $FAIL failed"
echo "============================================"
echo "Summary:"
for savedir in "$RESULT_ROOT"/*/; do
    name=$(basename "$savedir")
    log="$savedir/log.log"
    if [[ -f "$log" ]]; then
        last=$(grep "Total.*success" "$log" | tail -1 || echo "?")
    else
        last="(no log)"
    fi
    echo "  $name — $last"
done
