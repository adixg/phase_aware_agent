#!/bin/bash
# Replay FreeInference sessions against vLLM with a small, fixed KV pool, to check the replay simulator
# under memory pressure: first a calibration (prefill, decode and post-eviction cost per token), then
# the closed-loop replay. One server per job, so run one job per KV mode.
#
#   sbatch scripts/pressure_replay.sbatch                         # recompute: vLLM alone, 125k-token pool
#   MODE=lmcache sbatch scripts/pressure_replay.sbatch            # + LMCache CPU tier (CPU_GB, default 22)
#   MODE=lmcache CPU_GB=7 sbatch scripts/pressure_replay.sbatch   # a CPU tier the size of the GPU pool
#   POOL_TOKENS=400000 sbatch scripts/pressure_replay.sbatch      # roomy pool: the no-pressure reference
#   WORKLOAD=workloads/pressure_2026-08-16.json MAX_NUM_SEQS=8 sbatch ...
#   MODE=native VLLM_ARGS='--kv-transfer-config {...}' sbatch ...  # vLLM's own CPU offloading (set its flags)
#   CONDA_ENV=$HOME/scratch/conda-envs/paa-lmcache MODE=lmcache sbatch ...   # an env with lmcache installed
#   VENV=$HOME/scratch/venvs/paa-lmcache MODE=lmcache sbatch ...   # or a venv layered on the conda env
#                                                          # (python -m venv --system-site-packages)
#   sbatch --time=00:20:00 scripts/pressure_replay.sbatch --limit-sessions 4   # smoke test (args go to replay)
#
# The 125k-token pool is the scaled-down equivalent (workload tokens / 16) of the simulator's 2M-token
# pool; CPU_GB 22 is ~3.2x the pool's 7 GB (Qwen2.5-7B: 56 KB per token), the DGX H100 ratio.
# Submit from the repo root. Expects the env at CONDA_ENV (default ~/scratch/conda-envs/paa; for
# MODE=lmcache one with lmcache installed, matching its vLLM) and a Qwen2.5-7B-Instruct snapshot in
# $HF_HOME. Results go to results/traces/pressure-<jobid>_<mode>_pool<N>/.
#SBATCH --job-name=paa-pressure
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH -C "L40S"                  # one GPU type: the timings are compared with the simulator
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                  # room for the LMCache CPU tier
#SBATCH --time=01:00:00            # ~5 min startup + ~5 min calibration + ~20-30 min replay
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

MODE="${MODE:-recompute}"                     # recompute | lmcache | native
POOL_TOKENS="${POOL_TOKENS:-125000}"
CPU_GB="${CPU_GB:-22}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"             # requests vLLM runs at once (the simulator's slots)
WORKLOAD="${WORKLOAD:-workloads/pressure_2026-08-16.json}"
BLOCKS=$(( (POOL_TOKENS + 15) / 16 ))         # vLLM's KV block is 16 tokens
OUT="${TMPDIR:-/tmp}/paa-pressure-${SLURM_JOB_ID}"
NAME="pressure-${SLURM_JOB_ID}_${MODE}_pool${POOL_TOKENS}"
mkdir -p "$OUT" logs results/traces

command -v conda >/dev/null || module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-$HOME/scratch/conda-envs/paa}"
if [[ -n "${VENV:-}" ]]; then    # packages installed there (lmcache) take precedence over the conda env's
    source "$VENV/bin/activate"
fi
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}" HF_HUB_OFFLINE=1

DEFAULT_MODEL=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ 2>/dev/null | head -1 || true)
MODEL="${MODEL:-$DEFAULT_MODEL}"
[[ -n "$MODEL" ]] || { echo "no MODEL given and no Qwen2.5-7B-Instruct snapshot in $HF_HOME/hub"; exit 1; }
echo "node $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  mode $MODE  pool $POOL_TOKENS tokens ($BLOCKS blocks)"
echo "vllm $(python -c 'import vllm; print(vllm.__version__)')"

read -ra EXTRA_VLLM_ARGS <<< "${VLLM_ARGS:-}"
KV_ARGS=()
case "$MODE" in
    recompute) ;;
    lmcache)
        python -c 'import lmcache; print("lmcache", getattr(lmcache, "__version__", "?"))' \
            || { echo "MODE=lmcache needs lmcache in the env: pip install lmcache (a version matching vLLM)"; exit 1; }
        # LMCache settings by environment; check these names against the installed version's docs
        export LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE="$CPU_GB"
        KV_ARGS=(--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}') ;;
    native)
        [[ -n "${VLLM_ARGS:-}" ]] || { echo "MODE=native: pass vLLM's offloading flags in VLLM_ARGS (they differ by version)"; exit 1; } ;;
    *) echo "unknown MODE $MODE"; exit 1 ;;
esac

PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
VLOG="logs/vllm-${SLURM_JOB_ID}.log"
SERVE=(vllm serve "$MODEL" --served-model-name Qwen/Qwen2.5-7B-Instruct --port "$PORT"
       --max-num-seqs "$MAX_NUM_SEQS" --max-model-len 32768 --num-gpu-blocks-override "$BLOCKS"
       --gpu-memory-utilization 0.90 --enable-prompt-tokens-details --enable-prefix-caching
       --generation-config vllm "${KV_ARGS[@]}" "${EXTRA_VLLM_ARGS[@]}")
export PAA_VLLM_CMD="${SERVE[*]}" PAA_KV_MODE="$MODE" PAA_POOL_TOKENS="$POOL_TOKENS" PAA_CPU_GB="$CPU_GB" \
       PAA_MAX_NUM_SEQS="$MAX_NUM_SEQS"
echo "${SERVE[*]}"
"${SERVE[@]}" > "$VLOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

for _ in $(seq 120); do
    curl -sf "localhost:$PORT/health" >/dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited; see $VLOG"; exit 1; }
    sleep 5
done
curl -sf "localhost:$PORT/health" >/dev/null || { echo "vLLM not healthy after 10 min"; exit 1; }
sleep 1; grep -q '"GET /health' "$VLOG" || { echo "port $PORT is answered by another server, not this job's vLLM"; exit 1; }
echo "vLLM up on port $PORT"

URL="http://localhost:$PORT/v1"
python scripts/pressure_replay.py calibrate --base-url "$URL" --pool-tokens "$POOL_TOKENS" --out "$OUT" || STATUS=$?
python scripts/pressure_replay.py replay --base-url "$URL" --workload "$WORKLOAD" --out "$OUT" "$@" || STATUS=$?

cp "$VLOG" "$OUT/vllm.log"
cp -r "$OUT" "results/traces/$NAME"
echo "results: results/traces/$NAME"
exit "${STATUS:-0}"
