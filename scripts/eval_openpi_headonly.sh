#!/bin/bash
# Evaluate OpenPI head_only checkpoints (base camera only, no wrist, no tactile).
#
# Checkpoints:
#   /home/a25278/Workspaces/TactileWS/ckpts/pi05_insert_HDMI_60_epoch_headonly
#   /home/a25278/Workspaces/TactileWS/ckpts/pi05_insert_hole_60_epoch_headonly
#   /home/a25278/Workspaces/TactileWS/ckpts/pi05_lift_can_60_epoch_headonly
#
# Usage:
#   bash scripts/eval_openpi_headonly.sh --test          # 1 seed each, with video
#   bash scripts/eval_openpi_headonly.sh                 # 50 seeds each, no video
#   bash scripts/eval_openpi_headonly.sh --push --hf_repo user/repo

set -euo pipefail

PROJ_ROOT="/home/a25278/Workspaces/TactileWS/UniVTAC"
CKPT_BASE="/home/a25278/Workspaces/TactileWS/ckpts"
RESULT_BASE="$PROJ_ROOT/eval_result/OpenPI_headonly"
DEPLOY_YML="$PROJ_ROOT/policy/OpenPI/deploy_headonly.yml"

TEST_MODE=false
PUSH_HF=false
HF_REPO=""
MAX_SEED=49
START_SEED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)       TEST_MODE=true; MAX_SEED=0 ;;
        --push)       PUSH_HF=true ;;
        --hf_repo)    HF_REPO="$2"; shift ;;
        --max_seed)   MAX_SEED="$2";   shift ;;
        --start_seed) START_SEED="$2"; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
    shift
done

cd "$PROJ_ROOT"
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES

declare -A TASK_MAP=(
    ["pi05_insert_HDMI_60_epoch_headonly"]="insert_HDMI"
    ["pi05_insert_hole_60_epoch_headonly"]="insert_hole"
    ["pi05_lift_can_60_epoch_headonly"]="lift_can"
)

mkdir -p "$RESULT_BASE"

echo "============================================"
echo " OpenPI Head-Only Eval"
echo " Mode:   $( $TEST_MODE && echo 'TEST (1 seed)' || echo 'BATCH (50 seeds)' )"
echo " Tasks:  ${#TASK_MAP[@]}"
echo "============================================"

RESULTS_SUMMARY="/tmp/openpi_headonly_summary.txt"
echo "" > "$RESULTS_SUMMARY"

for ckpt_dir in "${!TASK_MAP[@]}"; do
    task="${TASK_MAP[$ckpt_dir]}"
    ckpt_path="$CKPT_BASE/$ckpt_dir"

    echo ""
    echo "===== $task ($ckpt_dir) ====="

    # Write temp deploy.yml pointing to this checkpoint
    cat > "$DEPLOY_YML" << EOF
policy_name: OpenPI
seed: 0
checkpoint_dir: $ckpt_path
tactile_mode: head_only
exec_horizon: 32
tokenizer_path: $CKPT_BASE/paligemma_tokenizer.model
instruction_type: seen
instuction_file: null
EOF

    outfile="/tmp/eval_openpi_${task}_headonly.txt"
    echo "  Starting..."
    t0=$(date +%s)

    if timeout 14400 python scripts/eval_policy.py "$task" demo OpenPI/deploy_headonly \
        --start_seed "$START_SEED" --max_seed "$MAX_SEED" \
        > "$outfile" 2>&1; then

        t1=$(date +%s)
        dt=$((t1 - t0))
        logsrc=$(ls -td "$PROJ_ROOT/eval_result/OpenPI/$task/deploy_headonly/"*/log.log 2>/dev/null | head -1)
        last_line=$(grep "Final Result" "$logsrc" 2>/dev/null | tail -1 || echo "?")
        echo "  Done in ${dt}s — $last_line"
        echo "$task: $last_line (${dt}s)" >> "$RESULTS_SUMMARY"

        savedir="$RESULT_BASE/${task}"
        mkdir -p "$savedir"
        cp "$outfile" "$savedir/output.txt"
        [[ -n "$logsrc" ]] && cp "$logsrc" "$savedir/log.log"

        if $TEST_MODE; then
            videodir=$(ls -td "$PROJ_ROOT/eval_result/OpenPI/$task/deploy_headonly/"*/video 2>/dev/null | head -1)
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
        echo "$task: FAILED (${dt}s)" >> "$RESULTS_SUMMARY"
    fi

    pkill -9 -f "python.*eval_policy" 2>/dev/null || true
    sleep 5
done

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
