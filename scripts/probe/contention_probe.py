#!/usr/bin/env python3
"""
contention_probe.py -- find the load region where the server actually queues.

Why this matters more than it looks: the phase-aware scheduling project compares
four ordering policies (FCFS, length-based, current-phase, current+next-phase).
Ordering policies can only differ when there is a QUEUE to order. If every run
sits in the flat, under-loaded region of the latency curve, all four policies
will produce statistically identical results and the project will have measured
nothing except that the server was idle.

So before committing to an experiment grid, sweep concurrency and find the knee
-- the point where mean latency starts climbing superlinearly. The real runs in
weeks 7-8 should operate at or above that knee.

Output is a JSON file plus a plain-text curve printed to stdout.
"""

import argparse
import json
import os
import statistics
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=64,
                   help="server concurrency cap. Keep FIXED across all four "
                        "policies in the real experiments -- this is the "
                        "'fixed resources' condition from proposal S2.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--concurrency", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64],
                   help="offered load levels to sweep")
    p.add_argument("--prompt-tokens", type=int, default=1024,
                   help="approximate prompt length, chosen to resemble an "
                        "agent turn carrying retrieved context")
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--replays", type=int, default=3,
                   help="repeats per concurrency level; the proposal requires "
                        "reporting run-to-run variability")
    p.add_argument("--out", default="contention.json")
    return p.parse_args()


def build_prompt(approx_tokens: int, tokenizer) -> str:
    """Build a prompt of roughly the requested token length."""
    filler = ("The scheduler observes the current phase of each agent session "
              "and estimates the phase that follows it. ")
    text = filler
    while len(tokenizer.encode(text)) < approx_tokens:
        text += filler
    # Trim back to target so every request is the same shape.
    ids = tokenizer.encode(text)[:approx_tokens]
    return tokenizer.decode(ids)


def main():
    args = parse_args()

    t_import = time.time()
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except Exception as e:
        print(f"FATAL: could not import serving stack: {e}", file=sys.stderr)
        return 2
    print(f"[probe] imports took {time.time() - t_import:.1f}s")

    print(f"[probe] loading tokenizer for {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"[probe] loading model onto GPU (this is the cold-start cost you "
          f"will pay every job -- note it for allocation planning)")
    t_load = time.time()
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="auto",
        enforce_eager=False,
    )
    load_s = time.time() - t_load
    print(f"[probe] model load took {load_s:.1f}s")

    prompt = build_prompt(args.prompt_tokens, tokenizer)
    # Fixed output length keeps this a clean queueing measurement rather than a
    # length-variance measurement. ignore_eos forces exactly N tokens.
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
    )

    # Warm up so CUDA graph capture and allocator behaviour do not land in the
    # first measured point.
    print("[probe] warming up")
    llm.generate([prompt] * 2, sp, use_tqdm=False)

    results = []
    for conc in args.concurrency:
        if conc > args.max_num_seqs:
            print(f"[probe] concurrency {conc} exceeds max_num_seqs "
                  f"{args.max_num_seqs} -- requests WILL queue in the server")
        latencies = []
        makespans = []
        for rep in range(args.replays):
            batch = [prompt] * conc
            t0 = time.time()
            outs = llm.generate(batch, sp, use_tqdm=False)
            makespan = time.time() - t0
            makespans.append(makespan)
            # vLLM's offline API returns all results together, so per-request
            # arrival/finish detail is limited; makespan per request is the
            # honest per-level number here. The real experiments should use the
            # async/server API to capture true per-call queueing time.
            latencies.append(makespan)
            print(f"  conc={conc:>3} rep={rep} makespan={makespan:.3f}s "
                  f"throughput={conc * args.output_tokens / makespan:.1f} tok/s")

        mean_ms = statistics.mean(makespans)
        row = {
            "concurrency": conc,
            "mean_makespan_s": mean_ms,
            "stdev_makespan_s": statistics.stdev(makespans) if len(makespans) > 1 else 0.0,
            "per_request_s": mean_ms / conc,
            "output_tok_per_s": conc * args.output_tokens / mean_ms,
            "replays": args.replays,
        }
        results.append(row)

    # ---- find the knee -------------------------------------------------
    # Normalise makespan by the single-request baseline. In the flat region
    # makespan barely grows with concurrency (the batch absorbs it). Past the
    # knee, makespan grows roughly linearly -- that is queueing.
    base = results[0]["mean_makespan_s"]
    knee = None
    for r in results:
        r["makespan_vs_single"] = r["mean_makespan_s"] / base
        # heuristic: knee = first level where makespan is >1.5x the single-
        # request makespan, i.e. the batch can no longer hide the added load
        if knee is None and r["makespan_vs_single"] > 1.5 and r["concurrency"] > 1:
            knee = r["concurrency"]

    print("\n" + "=" * 68)
    print("CONTENTION CURVE")
    print("=" * 68)
    print(f"{'conc':>6} {'makespan_s':>12} {'stdev':>8} {'per_req_s':>11} "
          f"{'tok/s':>9} {'x_single':>9}")
    for r in results:
        print(f"{r['concurrency']:>6} {r['mean_makespan_s']:>12.3f} "
              f"{r['stdev_makespan_s']:>8.3f} {r['per_request_s']:>11.3f} "
              f"{r['output_tok_per_s']:>9.1f} {r['makespan_vs_single']:>9.2f}")

    print("\n" + "-" * 68)
    if knee is None:
        print("VERDICT: no knee found in the swept range.")
        print("  The server absorbed every load level tested. At these settings")
        print("  a scheduling policy has almost nothing to reorder, and all four")
        print("  policies would likely tie.")
        print("  ACTION: raise --concurrency beyond max_num_seqs, lengthen")
        print("  prompts/outputs, or LOWER --max-num-seqs to tighten the server.")
    else:
        print(f"VERDICT: knee at roughly concurrency = {knee}.")
        print(f"  Run the weeks 7-8 comparative experiments AT OR ABOVE this")
        print(f"  level. Below it the server is under-loaded and the four")
        print(f"  policies will be indistinguishable.")
    print("-" * 68)

    payload = {
        "model": args.model,
        "gpu": os.environ.get("PROBE_GPU", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model_load_seconds": load_s,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "knee_concurrency": knee,
        "curve": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[probe] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
