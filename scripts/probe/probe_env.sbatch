#!/bin/bash
#SBATCH --job-name=probe-a100
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:a100:1          # A100 40/80GB. Overridden by submit_probe.sh.
                                   # Unlike the DSpark probe this project needs
                                   # NO FlashAttention-3 and NO Hopper feature,
                                   # so A100 / L40S / A40 are all valid targets.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # scheduler + agent sessions + tools are all
                                   # CPU-side and run concurrently with serving (w/ 16 cores)
#SBATCH --mem=96G
#SBATCH --time=00:30:00            # deliberately SHORT: short walltime is the
                                   # single biggest lever on queue wait, and the
                                   # point of this job is to measure that wait
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# NOTE: no `set -e`. This is a diagnostic probe: we want it to keep going and
# report EVERY missing piece in one run, not die on the first gap.
set -uo pipefail

cd "$SLURM_SUBMIT_DIR"

JOBID="${SLURM_JOB_ID}"
RES="results/probe-${JOBID}"
mkdir -p "$RES" .probe-state logs

# Record the exact moment execution began, so check_probe.sh can compute the
# true queue wait independent of whether sacct accounting is enabled.
date +%s > ".probe-state/${JOBID}.started"

SUMMARY="${RES}/SUMMARY.txt"
DETAIL="${RES}/detail.log"

# Every check writes a one-line PASS/WARN/FAIL verdict to the summary and its
# full output to the detail log.
PASS=0; WARN=0; FAIL=0
say()  { echo "$@" | tee -a "$SUMMARY"; }
note() { echo "$@" >> "$DETAIL"; }
verdict() { # verdict LEVEL "message"
    case "$1" in
        PASS) PASS=$((PASS+1));; WARN) WARN=$((WARN+1));; FAIL) FAIL=$((FAIL+1));;
    esac
    printf "  [%-4s] %s\n" "$1" "$2" | tee -a "$SUMMARY"
}
section() {
    say ""
    say "=============================================================="
    say "$1"
    say "=============================================================="
    note ""; note "########## $1 ##########"
}

: > "$SUMMARY"; : > "$DETAIL"

say "Phase-Aware Agent Scheduling -- allocation & environment probe"
say "job id      : ${JOBID}"
say "job name    : ${SLURM_JOB_NAME:-?}"
say "node        : $(hostname)"
say "started at  : $(date '+%Y-%m-%d %H:%M:%S %Z')"
say "submit dir  : ${SLURM_SUBMIT_DIR}"
say "requested   : ${SLURM_CPUS_PER_TASK:-?} cpus, ${SLURM_MEM_PER_NODE:-?}MB mem, ${SLURM_JOB_GRES:-?}"

# ---------------------------------------------------------------------------
# 1. GPU -- identity, memory, and whether it is EXCLUSIVELY ours
# ---------------------------------------------------------------------------
section "1. GPU"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    verdict FAIL "nvidia-smi not found -- no GPU visible on this node"
else
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,driver_version \
               --format=csv >> "$DETAIL" 2>&1
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used \
               --format=csv,noheader | sed 's/^/    /' | tee -a "$SUMMARY"

    GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    MEMTOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    MEMUSED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

    # bf16 needs Ampere or newer (compute cap >= 8.0). V100 (7.0) is a real
    # trap here: vLLM will run but fall back to fp16 with worse numerics and
    # much weaker kernel coverage.
    MAJOR="${CAP%%.*}"
    if [[ "$MAJOR" -ge 8 ]]; then
        verdict PASS "compute cap ${CAP} -- bf16 supported, vLLM fully supported"
    else
        verdict FAIL "compute cap ${CAP} is pre-Ampere. bf16 unavailable; request A100/L40S/A40 instead"
    fi

    # Memory sizing against the planned 7-8B bf16 model (~16GB weights).
    if   [[ "$MEMTOT" -ge 70000 ]]; then
        verdict WARN "${MEMTOT}MB VRAM -- ample, but an 8B model may NOT saturate it. You will need high concurrency (or a bigger model) to create queueing pressure at all"
    elif [[ "$MEMTOT" -ge 38000 ]]; then
        verdict PASS "${MEMTOT}MB VRAM -- ideal for 8B bf16 (~16GB weights, ~20GB KV cache)"
    elif [[ "$MEMTOT" -ge 22000 ]]; then
        verdict WARN "${MEMTOT}MB VRAM -- use a 4-bit AWQ/GPTQ 8B model, not bf16"
    else
        verdict FAIL "${MEMTOT}MB VRAM is too small for this project"
    fi

    # Someone else's memory on "our" GPU wrecks measurement validity.
    if [[ "$MEMUSED" -gt 1000 ]]; then
        verdict FAIL "${MEMUSED}MB already in use by another process -- GPU is SHARED. Latency measurements would be contaminated. Ask for exclusive GPU access"
    else
        verdict PASS "GPU appears idle/exclusive (${MEMUSED}MB in use)"
    fi
fi

# DCGM gives higher-rate, lower-overhead utilisation sampling than nvidia-smi
# polling -- relevant to the phase-level GPU/memory logging in proposal S3.
if command -v dcgmi >/dev/null 2>&1; then
    verdict PASS "dcgmi present -- use it for phase-level GPU utilisation sampling"
    dcgmi discovery -l >> "$DETAIL" 2>&1
else
    verdict WARN "dcgmi not found -- fall back to 'nvidia-smi dmon -s um -d 1' (coarser, ~1Hz)"
fi

# ---------------------------------------------------------------------------
# 2. CPU -- the quiet failure mode for this specific project
# ---------------------------------------------------------------------------
section "2. CPU (scheduler + agent sessions + tools all live here)"

NCPU_ALLOC="${SLURM_CPUS_PER_TASK:-0}"
NCPU_SEEN=$(nproc 2>/dev/null || echo 0)
AFFINITY=$(taskset -cp $$ 2>/dev/null | sed 's/.*: //')

note "$(lscpu 2>&1)"
say "    allocated cpus-per-task : ${NCPU_ALLOC}"
say "    nproc visible           : ${NCPU_SEEN}"
say "    cpu affinity mask       : ${AFFINITY:-unknown}"

if [[ "$NCPU_ALLOC" -ge 16 ]]; then
    verdict PASS "${NCPU_ALLOC} cores -- enough to pin server and scheduler to disjoint sets"
elif [[ "$NCPU_ALLOC" -ge 8 ]]; then
    verdict WARN "${NCPU_ALLOC} cores is workable but tight. CPU contention can masquerade as scheduler queueing delay and silently corrupt your headline metric"
else
    verdict FAIL "${NCPU_ALLOC} cores is too few -- request --cpus-per-task=16"
fi

# Core pinning is what keeps vLLM's tokenizer threads from stealing time from
# your scheduler process and polluting measured queue times.
if command -v taskset >/dev/null 2>&1; then
    verdict PASS "taskset available -- pin server and scheduler to disjoint core sets in the real runs"
else
    verdict WARN "taskset missing -- use cgroup cpusets or numactl for pinning"
fi

# ---------------------------------------------------------------------------
# 3. System memory
# ---------------------------------------------------------------------------
section "3. System RAM"

free -g | tee -a "$DETAIL" | sed 's/^/    /' | tee -a "$SUMMARY"
MEMGB=$(free -g | awk '/^Mem:/{print $2}')
if   [[ "$MEMGB" -ge 90 ]]; then verdict PASS "${MEMGB}GB visible"
elif [[ "$MEMGB" -ge 60 ]]; then verdict WARN "${MEMGB}GB -- fine unless your retrieval corpus/index is large"
else verdict FAIL "${MEMGB}GB -- too little for dozens of concurrent sessions plus an index"
fi

# ---------------------------------------------------------------------------
# 4. Storage -- weights, index, and TRACE LOGGING latency
# ---------------------------------------------------------------------------
section "4. Storage"

export HF_HOME="${HF_HOME:-$HOME/scratch/hf}"
say "    HF_HOME      : ${HF_HOME}"
say "    scratch quota:"
( quota -s 2>/dev/null || df -h "$HOME/scratch" 2>/dev/null || df -h . ) \
    | tee -a "$DETAIL" | sed 's/^/      /' | tee -a "$SUMMARY"

for d in "$HOME/scratch" "/tmp" "/local" "/scratch"; do
    [[ -d "$d" ]] || continue
    fstype=$(stat -f -c %T "$d" 2>/dev/null)
    note "  $d -> fstype=$fstype"
    say  "    $d : fstype=${fstype}"
done

# Write throughput matters twice: cold-start weight loads across many repeated
# runs, and per-event trace logging during the runs themselves.
TESTDIR="${TMPDIR:-/tmp}"
say "    write test in ${TESTDIR}:"
dd if=/dev/zero of="${TESTDIR}/probe_io_$$" bs=1M count=1024 2>&1 | tail -1 \
    | sed 's/^/      /' | tee -a "$SUMMARY"
rm -f "${TESTDIR}/probe_io_$$"

if [[ -d "${TMPDIR:-}" ]] || [[ -d /local ]]; then
    verdict PASS "node-local scratch exists -- write trace logs THERE, not to shared FS"
else
    verdict WARN "no obvious node-local scratch; logging to shared FS adds latency into your measurements"
fi

# ---------------------------------------------------------------------------
# 5. Network -- compute nodes are usually offline; models must be pre-staged
# ---------------------------------------------------------------------------
section "5. Network / model staging"

if timeout 8 curl -sSf -o /dev/null https://huggingface.co 2>/dev/null; then
    verdict PASS "compute node HAS outbound internet -- models can download in-job"
    NET_OK=1
else
    verdict WARN "no outbound internet (normal). Pre-download models from the LOGIN node with HF_HUB_OFFLINE unset, then set HF_HUB_OFFLINE=1 here"
    NET_OK=0
fi

say "    contents of HF cache:"
if [[ -d "$HF_HOME" ]]; then
    du -sh "$HF_HOME" 2>/dev/null | sed 's/^/      /' | tee -a "$SUMMARY"
    find "$HF_HOME" -maxdepth 3 -name "models--*" -type d 2>/dev/null \
        | sed 's#.*models--##; s#--#/#g; s/^/      /' | head -20 | tee -a "$SUMMARY"
else
    say "      (HF_HOME does not exist yet)"
fi

# The models worth checking for -- all strong tool-callers in the 7-8B class.
CANDIDATES=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "mistralai/Ministral-8B-Instruct-2410"
)
FOUND_MODEL=""
for m in "${CANDIDATES[@]}"; do
    cachename="models--${m//\//--}"
    if [[ -d "${HF_HOME}/hub/${cachename}" ]] || [[ -d "${HF_HOME}/${cachename}" ]]; then
        verdict PASS "model already staged: ${m}"
        [[ -z "$FOUND_MODEL" ]] && FOUND_MODEL="$m"
    else
        note "not staged: $m"
    fi
done
if [[ -z "$FOUND_MODEL" ]]; then
    verdict FAIL "no candidate model staged. From the LOGIN node run:
           export HF_HOME=${HF_HOME}
           huggingface-cli download Qwen/Qwen2.5-7B-Instruct"
fi

# ---------------------------------------------------------------------------
# 6. Python / serving stack
# ---------------------------------------------------------------------------
section "6. Software stack"

# conda activate does not work in a non-interactive batch shell until the shell
# hook is sourced -- same gotcha as your DSpark script.
if module load anaconda3 2>/dev/null; then
    note "module load anaconda3 ok"
else
    note "module load anaconda3 failed or modules unavailable"
fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
    if conda activate "$HOME/scratch/conda-envs/cs6222" 2>/dev/null; then
        verdict PASS "conda env cs6222 activated"
    else
        verdict WARN "could not activate ~/scratch/conda-envs/cs6222 -- create it before the real runs"
    fi
else
    verdict WARN "conda not on PATH"
fi

say "    python: $(command -v python || echo MISSING)"
python -c "import sys; print('    python version:', sys.version.split()[0])" 2>&1 | tee -a "$SUMMARY"

check_pkg() {
    python - "$1" <<'PY' 2>/dev/null
import importlib, sys
name = sys.argv[1]
try:
    m = importlib.import_module(name)
    print(getattr(m, "__version__", "present"))
except Exception:
    sys.exit(1)
PY
}

for pkg in torch vllm transformers; do
    if v=$(check_pkg "$pkg"); then
        verdict PASS "${pkg} ${v}"
    else
        verdict FAIL "${pkg} NOT importable"
    fi
done
for pkg in sentence_transformers faiss numpy pandas matplotlib aiohttp; do
    if v=$(check_pkg "$pkg"); then
        verdict PASS "${pkg} ${v}"
    else
        verdict WARN "${pkg} not importable (needed for retrieval tool / analysis)"
    fi
done

python -c "
import torch
print('    torch.cuda.is_available():', torch.cuda.is_available())
if torch.cuda.is_available():
    print('    device:', torch.cuda.get_device_name(0))
    print('    bf16 supported:', torch.cuda.is_bf16_supported())
" 2>&1 | tee -a "$SUMMARY"

# ---------------------------------------------------------------------------
# 7. vLLM smoke test + CONTENTION CALIBRATION
#    This is the scientifically important part: it checks whether this GPU,
#    at this model size, actually BUILDS A QUEUE. If it never queues, all four
#    of your policies will tie and the experiment is dead on arrival.
# ---------------------------------------------------------------------------
section "7. Serving smoke test + contention calibration"

MODEL="${MODEL:-$FOUND_MODEL}"
if [[ -z "$MODEL" ]]; then
    verdict WARN "skipping serving test -- no model staged"
elif ! check_pkg vllm >/dev/null; then
    verdict WARN "skipping serving test -- vllm not importable"
else
    say "    model under test: ${MODEL}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    [[ "$NET_OK" == "1" ]] && export HF_HUB_OFFLINE=0

    timeout 1200 python bench/contention_probe.py \
        --model "$MODEL" \
        --max-model-len 8192 \
        --max-num-seqs 64 \
        --concurrency 1 2 4 8 16 32 64 \
        --prompt-tokens 1024 \
        --output-tokens 128 \
        --out "${RES}/contention.json" 2>&1 | tee -a "$DETAIL" | tail -40 | tee -a "$SUMMARY"

    if [[ -f "${RES}/contention.json" ]]; then
        verdict PASS "contention calibration completed -- see ${RES}/contention.json"
    else
        verdict FAIL "contention calibration did not produce output (see detail.log)"
    fi
fi

# ---------------------------------------------------------------------------
# 8. Verdict
# ---------------------------------------------------------------------------
date +%s > ".probe-state/${JOBID}.ended"

section "8. VERDICT"
say "    PASS: ${PASS}    WARN: ${WARN}    FAIL: ${FAIL}"
say ""
if [[ "$FAIL" -eq 0 ]]; then
    say "    READY. This node configuration can host the project."
else
    say "    NOT READY. Resolve every [FAIL] above before requesting a long allocation."
fi
say ""
say "    full detail : ${DETAIL}"
say "    stdout log  : logs/${SLURM_JOB_NAME:-probe}-${JOBID}.out"
say "    calibration : ${RES}/contention.json"
say ""
say "finished at: $(date '+%Y-%m-%d %H:%M:%S %Z')"

echo
echo "########################################################################"
echo "# Read the summary with:  cat ${SUMMARY}"
echo "########################################################################"
