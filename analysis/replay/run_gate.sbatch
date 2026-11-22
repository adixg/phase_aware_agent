#!/bin/bash
# The oracle-gate sweeps on a PACE ICE CPU node (the replay is serial Python; its runs are
# independent, so it scales with cores and needs no GPU). Submit from the repo root:
#
#   sbatch analysis/replay/run_gate.sbatch                              # main sweep (class_l40s)
#   SWEEPS=class_l40s,prop_l40s sbatch analysis/replay/run_gate.sbatch
#   SWEEPS=all sbatch --time=02:00:00 analysis/replay/run_gate.sbatch   # all five sweeps
#   sbatch --cpus-per-task=32 analysis/replay/run_gate.sbatch           # more cores, finishes sooner
#   sbatch -A <account> analysis/replay/run_gate.sbatch                 # if your allocation needs one
#
# Expects the conda env at ~/scratch/conda-envs/paa with numpy, orjson and zstandard (compute nodes
# are offline: install on the login node, `pip install numpy orjson zstandard`) and the extracted
# traces in data/freeinference/compact/ (scripts/pace_sync.sh push-data copies them from a laptop).
# Results land in data/freeinference/gate/ and the log in logs/paa-gate-<jobid>.out;
# scripts/pace_sync.sh pull brings both back. A sweep rewrites its output file, so a second job
# running the same sweep replaces the first one's results.
#
# Runs are ~0.1-0.3 s each; the main sweep is 11,466 of them, ~21 runs/s on 6 cores of a laptop.
# WORKERS is set from --cpus-per-task, so ~16 cores should take a few minutes.
#SBATCH --job-name=paa-gate
#SBATCH --partition=ice-cpu        # unverified: confirm with `sinfo -s` on the login node
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G                  # ~0.3 GB per worker process, plus the loaded traces
#SBATCH --time=01:00:00            # generous for the main sweep; all five need --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/freeinference/gate

# sbatch copies the submitting shell's environment; if conda is already on PATH,
# reloading the module only prints a spurious "conda deactivate" error.
command -v conda >/dev/null || module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/scratch/conda-envs/paa"

python -c "import numpy, orjson, zstandard" 2>/dev/null || {
    echo "the paa env lacks numpy, orjson or zstandard; on the login node run: pip install numpy orjson zstandard" >&2
    exit 1
}
n=$(ls data/freeinference/compact/*.json 2>/dev/null | wc -l)
[[ $n -ge 14 ]] || { echo "only $n of 14 files in data/freeinference/compact; run scripts/pace_sync.sh push-data" >&2; exit 1; }

echo "node $(hostname)  cpus $SLURM_CPUS_PER_TASK  sweeps ${SWEEPS:-class_l40s}"
export SWEEPS="${SWEEPS:-class_l40s}" WORKERS="$SLURM_CPUS_PER_TASK" PY=python
analysis/replay/run_all.sh

wc -l data/freeinference/gate/*.jsonl
echo "done; on your laptop: scripts/pace_sync.sh pull"
