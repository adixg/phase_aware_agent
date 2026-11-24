"""Replay real agent sessions against vLLM under KV-memory pressure, to check the replay simulator
(analysis/replay/sim.py) where its results matter most.

FreeInference releases token counts and shared-prefix lengths but no text, so each request is sent
as token IDs: its prompt is the previous prompt's first `lcp` tokens followed by new random tokens,
which gives vLLM the same prefix-reuse pattern as the trace. Outputs are forced to the trace's length
(ignore_eos). Everything is scaled down by the same factor (default 16x): prompts and outputs in tokens,
and tool gaps and user replies in time, so the replay is a time-compressed copy of the full-size
workload in which prefill, decode, waits and KV memory keep their proportions (per-token costs are set
by the hardware). The KV pool is set with --num-gpu-blocks-override to the matching scaled size.

    # locally (needs data/freeinference/compact): a small workload file to commit
    python scripts/pressure_replay.py prepare --day 2026-08-16 --out workloads/pressure_2026-08-16.json
    # on PACE, against a running server (see pressure_replay.sbatch)
    python scripts/pressure_replay.py calibrate --base-url http://localhost:PORT/v1 --pool-tokens 125000 --out DIR
    python scripts/pressure_replay.py replay --base-url http://localhost:PORT/v1 --workload FILE --out DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics as st
import sys
import threading
import time
import urllib.request
from pathlib import Path

VOCAB = (1000, 100_000)      # token IDs drawn from here: ordinary tokens for Qwen2.5 and Llama-3 vocabularies
USER_REPLY_S = 5.0


# --- workload --------------------------------------------------------------------------------------

def prepare(args) -> None:
    """Sessions from one trace day, scaled to fit the model: prompt, shared prefix, output, gap after."""
    files = sorted(glob.glob(os.path.join(args.compact, f"*__{args.day}.json")))
    if not files:
        sys.exit(f"no compact files for {args.day} in {args.compact}")
    raw = [s for f in files for s in json.load(open(f))["sessions"]]
    raw.sort(key=lambda s: (s["start"], s["sid"]))
    limit = args.max_model_len - args.max_output
    gap_scale = args.gap_scale if args.gap_scale is not None else 1 / args.scale
    out = []
    for s in raw:
        reqs = s["requests"][:args.max_requests]
        if len(reqs) < args.min_requests:
            continue
        rows, prev = [], 0
        for i, r in enumerate(reqs):
            prompt = max(16, round(r["prompt"] / args.scale))
            lcp = 0 if i == 0 else min(round(r["lcp"] / args.scale), prev, prompt)
            gap = None
            if i + 1 < len(reqs):
                gap = (USER_REPLY_S if reqs[i + 1]["user"] else min(r["gap"] / 1000, args.max_gap)) * gap_scale
            completion = max(1, min(round(r["completion"] / args.scale), args.max_output))
            rows.append({"prompt": prompt, "lcp": lcp, "completion": completion,
                         "gap_after_s": gap, "user_turn": bool(r["user"]) or i == 0})
            prev = prompt
        if max(x["prompt"] for x in rows) > limit:
            continue
        out.append({"sid": s["sid"], "start_ms": s["start"], "requests": rows})
        if len(out) == args.sessions:
            break
    if args.target_load:
        # spread arrivals so GPU work / (slots x window) = target load, as the simulator's runs do;
        # work at the L40S rates the simulator was calibrated with, prefixes reused
        work = sum((140.0 + 0.046 * (r["prompt"] - r["lcp"]) + 22.4 * r["completion"]) / 1000
                   for s in out for r in s["requests"])
        args.window_s = round(work / (args.slots * args.target_load), 1)
    t0 = out[0]["start_ms"]
    span = max(s["start_ms"] for s in out) - t0 or 1
    for s in out:   # the trace's start order and spacing, squeezed into the arrival window
        s["arrival_s"] = round((s["start_ms"] - t0) / span * args.window_s, 3)
        del s["start_ms"]
    meta = {"source": "FreeInference agentic trace (harvardMadsys/freeinference_agentic_trace, CC BY 4.0)",
            "day": args.day, "token_scale": args.scale, "max_model_len": args.max_model_len,
            "max_output": args.max_output, "max_requests": args.max_requests, "max_gap_s": args.max_gap,
            "gap_scale": gap_scale, "user_reply_s": USER_REPLY_S * gap_scale, "window_s": args.window_s,
            "target_load": args.target_load, "slots": args.slots,
            "sessions": len(out), "requests": sum(len(s["requests"]) for s in out)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": meta, "sessions": out}, open(args.out, "w"))
    p = [r["prompt"] for s in out for r in s["requests"]]
    print(json.dumps(meta), f"prompt tokens p50 {st.median(p):.0f} max {max(p)}", sep="\n")


def prompts_for(session: dict) -> list[list[int]]:
    """Token IDs per request: the previous prompt's first `lcp` tokens, then new ones (deterministic)."""
    out, prev = [], []
    for i, r in enumerate(session["requests"]):
        rng = random.Random(f"{session['sid']}:{i}")
        new = [rng.randrange(*VOCAB) for _ in range(r["prompt"] - r["lcp"])]
        cur = prev[:r["lcp"]] + new
        out.append(cur)
        prev = cur
    return out


# --- talking to vLLM ---------------------------------------------------------------------------------

def complete(base_url: str, model: str, prompt: list[int], n_out: int, timeout: float = 900) -> dict:
    """One streamed completion: times to first token and end, and the server's token counts."""
    body = {"model": model, "prompt": prompt, "max_tokens": n_out, "min_tokens": n_out, "ignore_eos": True,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base_url.rstrip("/") + "/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first, usage = None, {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("choices") and chunk["choices"][0].get("text") and first is None:
                first = time.perf_counter()
            if chunk.get("usage"):
                usage = chunk["usage"]
    end = time.perf_counter()
    details = usage.get("prompt_tokens_details") or {}
    return {"ttft_s": (first or end) - t0, "total_s": end - t0, "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "cached_tokens": details.get("cached_tokens")}


def served_model(base_url: str) -> str:
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def calibrate(args) -> None:
    """Per-token prefill and decode time, and what a prompt costs again after its cache left the GPU
    (recompute without a CPU tier, a reload with one). Flushing sends unrelated prompts totalling
    twice the pool."""
    model = served_model(args.base_url)
    rng = random.Random(0)
    fresh = lambda n: [rng.randrange(*VOCAB) for _ in range(n)]
    complete(args.base_url, model, fresh(512), 8)                   # warm-up
    rows = []
    for n in args.lengths:
        p = fresh(n)
        cold = complete(args.base_url, model, p, 1)
        warm = complete(args.base_url, model, p, 1)
        flushed = 0
        while flushed < 2 * args.pool_tokens:
            k = min(n, args.max_model_len - 64)
            complete(args.base_url, model, fresh(k), 1)
            flushed += k
        evicted = complete(args.base_url, model, p, 1)
        rows.append({"tokens": n, "cold_ttft_s": cold["ttft_s"], "gpu_hit_ttft_s": warm["ttft_s"],
                     "after_eviction_ttft_s": evicted["ttft_s"], "cached_after_eviction": evicted["cached_tokens"]})
        print(json.dumps(rows[-1]), flush=True)
    dec = complete(args.base_url, model, fresh(256), 512)
    fit = lambda key: _slope([r["tokens"] for r in rows], [r[key] for r in rows])
    out = {"model": model, "pool_tokens": args.pool_tokens, "points": rows,
           "prefill_ms_per_token": 1000 * fit("cold_ttft_s")[0], "overhead_ms": 1000 * fit("cold_ttft_s")[1],
           "after_eviction_ms_per_token": 1000 * fit("after_eviction_ttft_s")[0],
           "decode_ms_per_token_alone": 1000 * (dec["total_s"] - dec["ttft_s"]) / 511}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(args.out) / "calibration.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "points"}))


def _slope(x, y):
    mx, my = st.mean(x), st.mean(y)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    return b, my - b * mx


def replay(args) -> None:
    """Every session on its own thread, closed loop: a request goes when the previous one has finished
    and its gap has passed. Each request's timings go to requests.jsonl."""
    work = json.load(open(args.workload))
    if args.limit_sessions:
        work["sessions"] = work["sessions"][:args.limit_sessions]
    model = served_model(args.base_url)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"model": model, "workload": str(args.workload), "workload_meta": work["meta"], "speedup": args.speedup,
                "limit_sessions": args.limit_sessions,
                "mode": os.environ.get("PAA_KV_MODE"), "pool_tokens": os.environ.get("PAA_POOL_TOKENS"),
                "cpu_gb": os.environ.get("PAA_CPU_GB"), "vllm_cmd": os.environ.get("PAA_VLLM_CMD"),
                "max_num_seqs": os.environ.get("PAA_MAX_NUM_SEQS"), "complete": False}
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)   # up front, so a run cut short still has one
    log = open(out / "requests.jsonl", "w")
    lock = threading.Lock()
    t0 = time.perf_counter() + 2.0
    poller = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from serving.instrumentation.vllm_metrics import MetricsPoller
        poller = MetricsPoller(args.base_url, out / "vllm_metrics.jsonl", interval_s=1.0).start()
    except Exception as e:                                           # metrics are a bonus, not required
        print("no metrics poller:", e)

    def run_session(s):
        prompts = prompts_for(s)
        time.sleep(max(0.0, t0 + s["arrival_s"] / args.speedup - time.perf_counter()))
        for i, (r, p) in enumerate(zip(s["requests"], prompts)):
            ready = time.perf_counter() - t0
            try:
                res = complete(args.base_url, model, p, r["completion"])
                err = None
            except Exception as e:
                res, err = {}, repr(e)
            done = time.perf_counter() - t0
            with lock:
                log.write(json.dumps({"sid": s["sid"], "i": i, "ready_s": ready, "end_s": done, "error": err,
                                      "prompt": r["prompt"], "lcp": r["lcp"], "completion": r["completion"],
                                      "gap_after_s": r["gap_after_s"], **res}) + "\n")
                log.flush()
            if r["gap_after_s"] is not None:
                time.sleep(r["gap_after_s"] / args.speedup)

    threads = [threading.Thread(target=run_session, args=(s,), daemon=True) for s in work["sessions"]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if poller:
        poller.stop()
    log.close()
    manifest["complete"] = True
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    rows = [json.loads(l) for l in open(out / "requests.jsonl")]
    errs = sum(r["error"] is not None for r in rows)
    print(f"{len(rows)} requests, {errs} errors, wall {max(r['end_s'] for r in rows):.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--compact", default="data/freeinference/compact")
    p.add_argument("--day", required=True)
    p.add_argument("--scale", type=float, default=16.0, help="trace tokens (prompt and output) per replayed token")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-output", type=int, default=1024)
    p.add_argument("--max-requests", type=int, default=30, help="requests kept per session")
    p.add_argument("--min-requests", type=int, default=3)
    p.add_argument("--max-gap", type=float, default=60.0, help="tool gaps longer than this (trace seconds) are cut")
    p.add_argument("--gap-scale", type=float, default=None, help="time scale of gaps and replies (default 1 / --scale)")
    p.add_argument("--sessions", type=int, default=48)
    p.add_argument("--window-s", type=float, default=600.0, help="arrivals spread over this many seconds")
    p.add_argument("--target-load", type=float, default=0.9, help="set the window from the work instead (0: use --window-s)")
    p.add_argument("--slots", type=int, default=8, help="the server's max-num-seqs, for --target-load")
    p.add_argument("--out", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--base-url", required=True)
    c.add_argument("--pool-tokens", type=int, required=True)
    c.add_argument("--lengths", type=lambda s: [int(x) for x in s.split(",")], default=[2048, 4096, 8192, 16384, 28672])
    c.add_argument("--max-model-len", type=int, default=32768)
    c.add_argument("--out", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--base-url", required=True)
    r.add_argument("--workload", required=True)
    r.add_argument("--speedup", type=float, default=1.0, help="divide arrivals and gaps by this")
    r.add_argument("--limit-sessions", type=int, default=0, help="replay only the first N sessions (smoke tests)")
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    {"prepare": prepare, "calibrate": calibrate, "replay": replay}[args.cmd](args)


if __name__ == "__main__":
    main()
