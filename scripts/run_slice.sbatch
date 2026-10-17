#!/bin/bash
# Serial AgenticRAGTracer run against a local vLLM server, on one PACE ICE GPU.
#
#   sbatch scripts/run_slice.sbatch                   # 20 tasks
#   LIMIT=100 sbatch scripts/run_slice.sbatch         # whole slice
#   LIMIT=all sbatch --time=01:30:00 scripts/run_slice.sbatch --tasks all   # whole dataset (1,305 tasks)
#   MODEL=/path/to/model/dir sbatch scripts/run_slice.sbatch
#   sbatch scripts/run_slice.sbatch --retriever embedding   # extra args go to run_slice.py
#   sbatch scripts/run_slice.sbatch --concurrency 8 --retrieve-delay lognormal:0.3,0.8
#   VLLM_ARGS="--num-gpu-blocks-override 1500" sbatch ...  # extra vllm serve flags (here: a small KV cache)
#   VLLM_BATCH_INVARIANT=1 sbatch ...        # VLLM_* env vars reach vllm serve; recorded in the manifest
#   sbatch -C A100-40GB scripts/run_slice.sbatch             # one GPU type only (timing runs)
#   sbatch --time=01:00:00 scripts/run_slice.sbatch         # give LIMIT=100 headroom (~12 min expected)
#
# Submit from the repo root. Expects the env at ~/scratch/conda-envs/paa, the
# model already downloaded (compute nodes are offline), and the dataset at
# data/agentic_rag_tracer/. MODEL defaults to the Llama-3.1-8B-Instruct
# snapshot directory in $HF_HOME (default ~/scratch/hf_cache). A directory is
# loaded as-is: offline loading by repo name refuses snapshots that skip
# optional repo files (LICENSE, original/, ...). Traces are written to node-local disk during the
# run and copied to results/traces/run-<jobid>_<settings>/ at the end, e.g.
# run-6160001_qwen2.5-7b-instruct_slice100_c8_lognormal0.3-0.8_emb-cpu.
#SBATCH --job-name=paa-slice
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
# GPU: any of these (all >= 40 GB), first free wins. Timings differ by GPU type,
# so for traces whose timings get compared, pin one: sbatch -C A100-40GB ...
#SBATCH -C "A100-40GB|A100-80GB|L40S|A40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                  # peak use so far ~12 GB (vLLM host side + retriever)
#SBATCH --time=00:20:00            # a 20-task run takes ~5 min incl. vLLM startup; short jobs start sooner
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

LIMIT="${LIMIT:-20}"
OUT="${TMPDIR:-/tmp}/paa-${SLURM_JOB_ID}"   # run_slice.py names the run folder inside it
mkdir -p "$OUT" logs results/traces

# sbatch copies the submitting shell's environment; if conda is already on PATH,
# reloading the module only prints a spurious "conda deactivate" error.
command -v conda >/dev/null || module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/scratch/conda-envs/paa"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}" HF_HUB_OFFLINE=1

DEFAULT_MODEL=$(ls -d "$HF_HOME"/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/ 2>/dev/null | head -1 || true)
MODEL="${MODEL:-$DEFAULT_MODEL}"
[[ -n "$MODEL" ]] || { echo "no MODEL given and no Llama-3.1-8B-Instruct snapshot in $HF_HOME/hub"; exit 1; }
# Serve a snapshot directory under its repo name, so traces and the manifest
# say meta-llama/Llama-3.1-8B-Instruct rather than a cache path.
SERVED="$MODEL"
if [[ -d "$MODEL" ]]; then
    repo=$(sed -n 's#.*/models--\([^/]*\)/snapshots/.*#\1#p' <<< "$MODEL")
    [[ -n "$repo" ]] && SERVED="${repo/--//}"
fi

echo "node $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  model $SERVED ($MODEL)"

# localhost is shared by every job on the node: with a fixed port, a second job's
# health check and requests silently reach the first job's server. Take a free
# port, and check below that the server answering is this job's own.
PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
VLOG="logs/vllm-${SLURM_JOB_ID}.log"
read -ra EXTRA_VLLM_ARGS <<< "${VLLM_ARGS:-}"
SERVE=(vllm serve "$MODEL" --served-model-name "$SERVED" --port "$PORT" --max-num-seqs 64
       --gpu-memory-utilization 0.90 --enable-prompt-tokens-details --generation-config vllm
       "${EXTRA_VLLM_ARGS[@]}")
export PAA_VLLM_CMD="${SERVE[*]}" PAA_VLLM_ARGS="${VLLM_ARGS:-}"   # recorded by run_slice.py
echo "${SERVE[*]}"
"${SERVE[@]}" > "$VLOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

for _ in $(seq 120); do                     # up to 10 min for weights + CUDA graphs
    curl -sf "localhost:$PORT/health" >/dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited; see $VLOG"; exit 1; }
    sleep 5
done
curl -sf "localhost:$PORT/health" >/dev/null || { echo "vLLM not healthy after 10 min"; exit 1; }
sleep 1; grep -q '"GET /health' "$VLOG" || { echo "port $PORT is answered by another server, not this job's vLLM"; exit 1; }
echo "vLLM up on port $PORT"

LIMIT_ARGS=(--limit "$LIMIT"); [[ "$LIMIT" == all ]] && LIMIT_ARGS=()   # LIMIT=all: every task
python scripts/run_slice.py --llm vllm --base-url "http://localhost:$PORT/v1" "${LIMIT_ARGS[@]}" --run-prefix "run-${SLURM_JOB_ID}" --out-dir "$OUT" "$@" || STATUS=$?

for d in "$OUT"/*/; do
    [[ -d "$d" ]] || continue   # no run folder: run_slice.py failed before starting
    cp -r "$d" results/traces/
    echo "traces: results/traces/$(basename "$d")"
done
exit "${STATUS:-0}"
