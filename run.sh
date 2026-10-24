#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

python_bin="${PYTHON:-python}"
# Base-offset PE: On/Off
base_offset_pe="${BASE_OFFSET_PE:-on}"
case "${base_offset_pe,,}" in
    on|true|1) bits=4 ;;
    off|false|0) bits=0 ;;
    *) printf 'BASE_OFFSET_PE must be on or off\n' >&2; exit 2 ;;
esac
offload="${OFFLOAD:-sequential}"
prompt="$(cat "${PROMPT_FILE:-prompt.txt}")"
export PYTHONDONTWRITEBYTECODE=1

bash build.sh
"$python_bin" generate.py \
    --model_id "${MODEL_ID:-tencent/HunyuanVideo}" \
    --model_revision "${MODEL_REVISION-refs/pr/18}" \
    --device "${DEVICE:-cuda:0}" --offload "$offload" \
    --seed 0 --height 480 --width 720 --num_frames 17 \
    --prompt "$prompt" --num_inference_steps 50 \
    --num_q_centroids 100 --num_k_centroids 500 \
    --top_p_kmeans 0.9 --min_kc_ratio 0.10 \
    --kmeans_iter_init 50 --kmeans_iter_step 2 --zero_step_kmeans_init \
    --first_times_fp 0.04 --first_layers_fp 0.0 \
    --base_offset_bits "$bits" "$@"
