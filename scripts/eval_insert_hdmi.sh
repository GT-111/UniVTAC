#!/bin/bash
# Evaluate insert_HDMI with both ACT variants and push to Hugging Face.
#
# Variants:
#   vision_only → ACT (train_config_vision, no tactile backbone)
#   univtac    → ACT (train_config,       tactile backbone)
#
# Usage:
#   bash scripts/eval_insert_hdmi.sh --test          # 1 seed each, with video
#   bash scripts/eval_insert_hdmi.sh                 # 100 seeds each, no video
#   bash scripts/eval_insert_hdmi.sh --push --hf_repo user/repo  # + push to HF

set -euo pipefail

PROJ_ROOT="/home/a25278/Workspaces/TactileWS/UniVTAC"
CKPT_ROOT="/home/a25278/Workspaces/TactileWS/ckpts/act/insert_HDMI"
ACT_CKPT_BASE="$PROJ_ROOT/policy/ACT/act_ckpt"
RESULT_BASE="$PROJ_ROOT/eval_result/insert_HDMI_batch"

TEST_MODE=false
PUSH_HF=false
HF_REPO=""
MAX_SEED=99
START_SEED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)       TEST_MODE=true; MAX_SEED=0 ;;
        --push)       PUSH_HF=true ;;
        --hf_repo)    HF_REPO="$2"; shift ;;
        --max_seed)   MAX_SEED="$2"; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
    shift
done

cd "$PROJ_ROOT"
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES

declare -a VARIANTS=(
    "vision_only|vision_only|train_config_vision"
    "univtac|univtac|train_config"
)

mkdir -p "$RESULT_BASE"

echo "============================================"
echo " insert_HDMI Eval"
echo " Mode:   $( $TEST_MODE && echo 'TEST (1 seed)' || echo 'BATCH (100 seeds)' )"
echo " Config: demo"
echo "============================================"

RESULTS_SUMMARY="/tmp/insert_hdmi_summary.txt"
echo "" > "$RESULTS_SUMMARY"

for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r label ckpt_subdir train_cfg <<< "$entry"

    echo ""
    echo "===== $label ($train_cfg) ====="

    # Symlink checkpoint
    target="$ACT_CKPT_BASE/act-insert_HDMI/demo-50/${train_cfg}"
    mkdir -p "$target"
    rm -f "$target/policy_best.ckpt" "$target/dataset_stats.pkl"
    ln -sf "$CKPT_ROOT/$ckpt_subdir/policy_last.ckpt"  "$target/policy_best.ckpt"
    ln -sf "$CKPT_ROOT/$ckpt_subdir/dataset_stats.pkl" "$target/dataset_stats.pkl"

    # Run eval
    outfile="/tmp/eval_insert_hdmi_${label}.txt"
    export TRAIN_CONFIG="$train_cfg"

    echo "  Starting..."
    t0=$(date +%s)

    if timeout 14400 python scripts/eval_policy.py insert_HDMI demo ACT/deploy \
        --start_seed "$START_SEED" --max_seed "$MAX_SEED" \
        > "$outfile" 2>&1; then

        t1=$(date +%s)
        dt=$((t1 - t0))
        logsrc=$(ls -td "$PROJ_ROOT/eval_result/ACT/insert_HDMI/deploy/"*/log.log 2>/dev/null | head -1)
        last_line=$(grep "Final Result" "$logsrc" 2>/dev/null | tail -1 || echo "?")
        echo "  Done in ${dt}s — $last_line"
        echo "$label: $last_line (${dt}s)" >> "$RESULTS_SUMMARY"

        savedir="$RESULT_BASE/${label}"
        mkdir -p "$savedir"
        cp "$outfile" "$savedir/output.txt"
        [[ -n "$logsrc" ]] && cp "$logsrc" "$savedir/log.log"

        if $TEST_MODE; then
            videodir=$(ls -td "$PROJ_ROOT/eval_result/ACT/insert_HDMI/deploy/"*/video 2>/dev/null | head -1)
            if [[ -n "$videodir" ]]; then
                cp -r "$videodir" "$savedir/video"
                echo "  Video: $savedir/video"
            fi
        fi
    else
        t1=$(date +%s)
        dt=$((t1 - t0))
        echo "  FAILED after ${dt}s"
        grep -E "Traceback|Error:" "$outfile" | head -5
        echo "$label: FAILED (${dt}s)" >> "$RESULTS_SUMMARY"
    fi

    pkill -9 -f "python.*eval_policy" 2>/dev/null || true
    sleep 5
done

# Cleanup stale config
rm -f "$PROJ_ROOT/task_config/_batch_insert_hdmi.yml"

echo ""
echo "============================================"
echo " RESULTS"
echo "============================================"
cat "$RESULTS_SUMMARY"

if $PUSH_HF; then
    [[ -z "$HF_REPO" ]] && { echo "ERROR: --hf_repo required"; exit 1; }
    echo "Pushing to $HF_REPO ..."
    HF_DIR="/tmp/hf_upload_$$"
    mkdir -p "$HF_DIR"
    cp -r "$RESULT_BASE"/* "$HF_DIR/"
    cp "$RESULTS_SUMMARY" "$HF_DIR/summary.txt"
    cd "$HF_DIR"
    huggingface-cli upload "$HF_REPO" . . --repo-type=dataset || echo "HF upload failed"
    rm -rf "$HF_DIR"
fi

echo "Local: $RESULT_BASE"
