#!/bin/bash
# Serial AgenticRAGTracer run against a local vLLM server, on one PACE ICE GPU.
#
#   sbatch scripts/run_slice.sbatch                   # 20 tasks
#   LIMIT=100 sbatch scripts/run_slice.sbatch         # whole slice
#   LIMIT=all sbatch --time=01:30:00 scripts/run_slice.sbatch --tasks all   # whole dataset (1,305 tasks)
#   MODEL=/path/to/model/dir sbatch scripts/run_slice.sbatch
#   sbatch scripts/run_slice.sbatch --retriever embedding   # extra args go to run_slice.py
#   sbatch scripts/run_slice.sbatch --concurrency 8 --retrieve-delay lognormal:0.3,0.8
#   sbatch -C A100-40GB scripts/run_slice.sbatch             # one GPU type only (timing runs)
#   sbatch --time=01:00:00 scripts/run_slice.sbatch         # give LIMIT=100 headroom (~12 min expected)
#
# Submit from the repo root. Expects the env at ~/scratch/conda-envs/paa, the
# model already downloaded (compute nodes are offline), and the dataset at
# data/agentic_rag_tracer/. MODEL defaults to the Llama-3.1-8B-Instruct
# snapshot directory in $HF_HOME (default ~/scratch/hf_cache). A directory is
# loaded as-is: offline loading by repo name refuses snapshots that skip
# optional repo files (LICENSE, original/, ...). Traces are written to node-local disk during the
# run and copied to results/traces/run-<jobid>/ at the end (the manifest's
# "mode" says serial or concurrent).
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
RUN_ID="run-${SLURM_JOB_ID}"
LOCAL="${TMPDIR:-/tmp}/${RUN_ID}"
mkdir -p "$LOCAL" logs results/traces

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

vllm serve "$MODEL" --served-model-name "$SERVED" --port 8000 --max-num-seqs 64 --gpu-memory-utilization 0.90 \
    --enable-prompt-tokens-details \
    --generation-config vllm \
    > "logs/vllm-${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

for _ in $(seq 120); do                     # up to 10 min for weights + CUDA graphs
    curl -sf localhost:8000/health >/dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited; see logs/vllm-${SLURM_JOB_ID}.log"; exit 1; }
    sleep 5
done
curl -sf localhost:8000/health >/dev/null || { echo "vLLM not healthy after 10 min"; exit 1; }

LIMIT_ARGS=(--limit "$LIMIT"); [[ "$LIMIT" == all ]] && LIMIT_ARGS=()   # LIMIT=all: every task
python scripts/run_slice.py --llm vllm "${LIMIT_ARGS[@]}" --run-id "$RUN_ID" --out-dir "${LOCAL%/*}" "$@" || STATUS=$?

cp -r "$LOCAL" results/traces/
echo "traces: results/traces/${RUN_ID}"
exit "${STATUS:-0}"
