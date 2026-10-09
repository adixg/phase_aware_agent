#!/bin/bash
# Serial AgenticRAGTracer run against a local vLLM server, on one PACE ICE GPU.
#
#   sbatch scripts/run_slice.sbatch                   # 20 tasks
#   LIMIT=100 sbatch scripts/run_slice.sbatch         # whole slice
#   MODEL=Qwen/Qwen2.5-3B-Instruct sbatch scripts/run_slice.sbatch
#
# Submit from the repo root. Expects the env at ~/scratch/conda-envs/paa, the
# model already in $HF_HOME (compute nodes are offline), and the dataset at
# data/agentic_rag_tracer/. Traces are written to node-local disk during the
# run and copied to results/traces/<run_id>/ at the end.
#SBATCH --job-name=paa-slice
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LIMIT="${LIMIT:-20}"
RUN_ID="serial-${SLURM_JOB_ID}"
LOCAL="${TMPDIR:-/tmp}/${RUN_ID}"
mkdir -p "$LOCAL" logs results/traces

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/scratch/conda-envs/paa"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf}" HF_HUB_OFFLINE=1

echo "node $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  model $MODEL"

vllm serve "$MODEL" --port 8000 --max-num-seqs 64 --gpu-memory-utilization 0.90 \
    > "logs/vllm-${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

for _ in $(seq 120); do                     # up to 10 min for weights + CUDA graphs
    curl -sf localhost:8000/health >/dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited; see logs/vllm-${SLURM_JOB_ID}.log"; exit 1; }
    sleep 5
done
curl -sf localhost:8000/health >/dev/null || { echo "vLLM not healthy after 10 min"; exit 1; }

python scripts/run_slice.py --llm vllm --limit "$LIMIT" --run-id "$RUN_ID" --out-dir "${LOCAL%/*}" || STATUS=$?

cp -r "$LOCAL" results/traces/
echo "traces: results/traces/${RUN_ID}"
exit "${STATUS:-0}"
