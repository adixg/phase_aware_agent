#!/bin/zsh
# The oracle-gate sweeps, from the repo root. Needs a Python with numpy, orjson and zstandard
# (PY=..., default python); results land in data/freeinference/gate/ (gitignored).
#
#   hf download harvardMadsys/freeinference_agentic_trace --type dataset --local-dir data/freeinference \
#       --include 'release/2026-08-23_2026-08-29/traces/*' --include 'release/2026-08-30_2026-09-05/traces/*'
#   analysis/replay/run_all.sh
#   python -m analysis.replay.report data/freeinference/gate/*.jsonl
#   python -m analysis.replay.breakdown data/freeinference/gate
set -e
export PYTHONPATH=.
PY=${PY:-python}
OUT=data/freeinference/gate
mkdir -p data/freeinference/compact $OUT

# 2026-08-29 appears in both weeks; the 4-session fragment in the second is skipped
for f in data/freeinference/release/*/traces/*.jsonl.zst; do
  d=$(basename $f .jsonl.zst)
  [[ $f == *2026-08-30_2026-09-05/traces/2026-08-29* ]] && continue
  [[ -f data/freeinference/compact/$d.json ]] || $PY -W ignore analysis/replay/extract.py $f data/freeinference/compact/$d.json
done

KV=0,1000000,2000000,4000000,8000000,16000000
GRID=(--loads 0.7,0.9,1.1 --kv $KV --workers ${WORKERS:-14})
$PY analysis/replay/run_gate.py --out $OUT/class_l40s.jsonl --profiles l40s --tight 0.5,1,2 $GRID
$PY analysis/replay/run_gate.py --out $OUT/prop_l40s.jsonl --profiles l40s --mode proportional --tight 1.2,1.5,2,3 $GRID
$PY analysis/replay/run_gate.py --out $OUT/class_offload.jsonl --profiles l40s_offload --tight 0.5,1,2 $GRID
$PY analysis/replay/run_gate.py --out $OUT/class_trace.jsonl --profiles trace --tight 0.5,1,2 $GRID
$PY analysis/replay/run_gate.py --out $OUT/class_l40s_slots16.jsonl --profiles l40s --slots 16 --tight 0.5,1,2 $GRID
