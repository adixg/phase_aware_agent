#!/usr/bin/env bash
# The oracle-gate sweeps, from the repo root. Needs a Python with numpy, orjson and zstandard
# (PY=..., default python); results land in data/freeinference/gate/ (gitignored).
#
#   hf download harvardMadsys/freeinference_agentic_trace --type dataset --local-dir data/freeinference \
#       --include 'release/2026-08-23_2026-08-29/traces/*' --include 'release/2026-08-30_2026-09-05/traces/*'
#   analysis/replay/run_all.sh
#   python -m analysis.replay.report data/freeinference/gate/*.jsonl
#   python -m analysis.replay.breakdown data/freeinference/gate
#
# Cost: measured on an 8-core M2 with 6 workers, ~21 runs/s. All five sweeps are ~48,000 runs, about
# 40 minutes; the main one (class_l40s, 11,466 runs with the lookahead-depth policies) about 9.
# Pick sweeps with SWEEPS:
#
#   SWEEPS=class_l40s analysis/replay/run_all.sh                 # the main sweep only
#   SWEEPS=class_l40s,prop_l40s WORKERS=4 analysis/replay/run_all.sh
#
# SWEEPS is a comma list of: class_l40s prop_l40s class_offload class_trace class_l40s_slots16 (default: all)
# or forecast_l40s (opt-in, ~3 min: Andrew's admission forecaster on the validation days Aug 30-Sep 1 only, so
# the predictor's held-out split Sep 2-5 stays untouched; report it with report.py on its own file). WORKERS defaults to the cores Slurm allocated, else the core count minus two.
# DRY_RUN=1 prints the commands instead of running them. Plain bash, so it also runs on the cluster:
# see analysis/replay/run_gate.sbatch.
set -e
export PYTHONPATH=.
PY=${PY:-python}
SWEEPS=${SWEEPS:-all}
CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
if [[ -n ${SLURM_CPUS_PER_TASK:-} ]]; then DEFAULT_WORKERS=$SLURM_CPUS_PER_TASK   # nproc would show the whole node
else DEFAULT_WORKERS=$(( CORES > 3 ? CORES - 2 : 1 )); fi
WORKERS=${WORKERS:-$DEFAULT_WORKERS}
RUN=""; [[ -n ${DRY_RUN:-} ]] && RUN=echo
OUT=${OUT:-data/freeinference/gate}
mkdir -p data/freeinference/compact $OUT

want() { [[ ",$SWEEPS," == *",all,"* || ",$SWEEPS," == *",$1,"* ]]; }
IFS=, read -ra REQUESTED <<< "$SWEEPS"
for s in "${REQUESTED[@]}"; do
  case $s in
    all|class_l40s|prop_l40s|class_offload|class_trace|class_l40s_slots16|forecast_l40s) ;;
    *) echo "unknown sweep: $s" >&2; exit 2 ;;
  esac
done

# 2026-08-29 appears in both weeks; the 4-session fragment in the second is skipped
shopt -s nullglob     # no release/ (only the compact files were copied over): nothing to extract
for f in data/freeinference/release/*/traces/*.jsonl.zst; do
  d=$(basename "$f" .jsonl.zst)
  [[ $f == *2026-08-30_2026-09-05/traces/2026-08-29* ]] && continue
  [[ -f data/freeinference/compact/$d.json ]] || $RUN $PY -W ignore analysis/replay/extract.py "$f" "data/freeinference/compact/$d.json"
done
for d in 2026-08-{23..29} 2026-08-30 2026-08-31 2026-09-0{1..5}; do
  [[ -f data/freeinference/compact/$d.json ]] || [[ -n $RUN ]] || { echo "missing data/freeinference/compact/$d.json" >&2; exit 1; }
done

KV=0,1000000,2000000,4000000,8000000,16000000
GRID=(--loads 0.7,0.9,1.1 --kv $KV --workers $WORKERS)
echo "sweeps: $SWEEPS, workers: $WORKERS"
# the lookahead-depth policies (--lookahead) run in the main sweep only; the rest are sensitivity checks
want class_l40s && $RUN $PY analysis/replay/run_gate.py --out $OUT/class_l40s.jsonl --profiles l40s --tight 0.5,1,2 --lookahead "${GRID[@]}"
want prop_l40s && $RUN $PY analysis/replay/run_gate.py --out $OUT/prop_l40s.jsonl --profiles l40s --mode proportional --tight 1.2,1.5,2,3 "${GRID[@]}"
want class_offload && $RUN $PY analysis/replay/run_gate.py --out $OUT/class_offload.jsonl --profiles l40s_offload --tight 0.5,1,2 "${GRID[@]}"
want class_trace && $RUN $PY analysis/replay/run_gate.py --out $OUT/class_trace.jsonl --profiles trace --tight 0.5,1,2 "${GRID[@]}"
want class_l40s_slots16 && $RUN $PY analysis/replay/run_gate.py --out $OUT/class_l40s_slots16.jsonl --profiles l40s --slots 16 --tight 0.5,1,2 "${GRID[@]}"
FORECAST_POLICIES=las,edf,llf_cheap,srpt_cheap,llf_calls,srpt_calls,llf_oracle,srpt_oracle,psrpt_cheap,psrpt_calls,psrpt_oracle,llf_predfirst,llf_predall,srpt_predfirst,srpt_predall,psrpt_predfirst,psrpt_predall
[[ ",$SWEEPS," == *",forecast_l40s,"* ]] && $RUN $PY analysis/replay/run_gate.py --out $OUT/forecast_l40s.jsonl --profiles l40s --tight 0.5,1,2 --days 2026-08-30,2026-08-31,2026-09-01 --forecast --only $FORECAST_POLICIES "${GRID[@]}"
true
