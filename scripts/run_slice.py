"""Run AgenticRAGTracer tasks against one LLM server and trace each job.

    python scripts/run_slice.py --llm vllm --limit 20
    python scripts/run_slice.py --llm vllm --concurrency 8 --retrieve-delay lognormal:0.3,0.8
    python scripts/run_slice.py --llm mock --limit 3      # plumbing check, no server

Builds the retriever once over the whole slice's pooled passages (so retrieval
difficulty doesn't depend on --limit), then runs each task through `run_one`.
Traces go to <out-dir>/<run-id>/<job_id>.jsonl next to a manifest.json that
records the run's settings, host and git commit. A job that raises (e.g. the
server dies) is logged in the manifest and the run continues.

--concurrency N keeps N jobs in flight (closed loop: a worker starts the next
task as soon as its job ends), so the server batches whichever jobs are in an
LLM call. --retrieve-workers K caps retrieves running at once at K (a pool of
K search workers): the rest wait, traced as WAIT spans; --llm-slots M does the
same for LLM calls, queueing them client side. vLLM streams its replies, so
each LLM span also splits into ttft_s (queued inside vLLM + prefill) and
decode_s. With --llm vllm, the server's /metrics are sampled to
server/vllm_metrics.jsonl (a subfolder, so trace loaders that glob *.jsonl
see only job traces).

--arrival-rate R switches to open loop: jobs arrive at seeded random times, R per
second on average (Poisson; --arrival-cv > 1 makes arrivals burstier), and each
starts at its arrival whatever else is running. Job completion time is then
end_ts - arrival_ts. Jobs run in a seeded shuffled order (the dataset is sorted
2-hop, 3-hop, 4-hop, so its own order would make load climb through a run).

Server settings the script can't see (vllm serve flags, VLLM_* env vars such as
VLLM_BATCH_INVARIANT) are read from the environment the sbatch script sets up
(PAA_VLLM_ARGS) and recorded in the manifest and the run name.

The run directory is what groups jobs that ran together; open-loop arrival
times are in the manifest's job entries.
"""

from __future__ import annotations

import argparse
import os
import json
import random
import socket
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent.delay import job_delay, parse_delay
from agent.llm_backend import MockLLM, VLLMBackend, react_system_prompt
from agent.retriever import DEFAULT_MODEL, EmbeddingRetriever, LocalRetriever, SentenceTransformerEmbedder
from agent.run_one import mock_script, run_one
from agent.workloads import get_workload, pool_documents
from serving.instrumentation.vllm_metrics import MetricsPoller


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_name() -> str | None:
    """GPU this run's server shares the node with; timings only compare within one type."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        return out.splitlines()[0].strip() or None
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


class LockedRetriever:
    """Serializes retrieve() across job threads: the embedding model isn't
    guaranteed thread-safe. Time spent waiting lands in the RETRIEVE span."""

    def __init__(self, inner):
        self.inner = inner
        self._lock = threading.Lock()

    def retrieve(self, query: str, top_k: int = 1):
        with self._lock:
            return self.inner.retrieve(query, top_k=top_k)


def run_label(args, model: str | None, num_tasks: int) -> str:
    """The settings that tell runs apart, for the run folder name, e.g.
    qwen2.5-7b-instruct_slice100_c8_lognormal0.3-0.8_emb-cpu. The manifest has the full record."""
    name = (model or args.llm).rstrip("/").rsplit("/", 1)[-1].lower()
    delay = args.retrieve_delay.replace(":", "").replace(",", "-") if args.retrieve_delay else "nodelay"
    if delay[0].isdigit():
        delay = f"delay{delay}"
    retriever = f"emb-{args.embedding_device}" if args.retriever == "embedding" else args.retriever
    arrivals = f"c{args.concurrency}"
    if args.arrival_rate:
        arrivals = f"poisson{args.arrival_rate:g}" if args.arrival_cv == 1 else \
            f"arr{args.arrival_rate:g}cv{args.arrival_cv:g}"
    parts = [name, f"{args.tasks}{num_tasks}", arrivals, delay, retriever]
    if args.retrieve_workers:
        parts.insert(4, f"w{args.retrieve_workers}")
    if args.llm_slots:
        parts.insert(4, f"llm{args.llm_slots}")
    if args.repetition_penalty != 1.0:
        parts.append(f"rp{args.repetition_penalty:g}")
    if os.environ.get("PAA_MAX_NUM_SEQS", "64") != "64":   # vLLM's per-step cap, if not the default
        parts.append(f"seqs{os.environ['PAA_MAX_NUM_SEQS']}")
    # extra `vllm serve` flags: "--num-gpu-blocks-override 1500 --enforce-eager"
    # -> num-gpu-blocks-override1500_enforce-eager
    takes_value = False
    for token in os.environ.get("PAA_VLLM_ARGS", "").split():
        if token.startswith("--"):
            parts.append(token[2:])
            takes_value = True
        elif takes_value:
            parts[-1] += token
            takes_value = False
    if os.environ.get("VLLM_BATCH_INVARIANT") == "1":
        parts.append("batchinv")
    return "_".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="root of fetched datasets (default: ./data)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", choices=["slice", "all"], default="slice",
                   help="slice: the 100-task reduced slice (skewed to 3/4-hop); all: every task in the "
                        "dataset. The retriever corpus is the pooled passages of the chosen set.")
    p.add_argument("--limit", type=int, default=None,
                   help="run a seeded random subset of the chosen tasks (keeps the hop mix); default: all")
    p.add_argument("--llm", choices=["vllm", "mock"], default="vllm")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default=None, help="default: whatever the server is serving")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="1.0 = off. A penalty also discourages copying names/dates from passages")
    p.add_argument("--max-steps", type=int, default=10,
                   help="generations per job; 4-hop questions need >= 5 even when every search hits")
    p.add_argument("--no-thinking", action="store_true",
                   help="turn off a hybrid model's <think> block (Qwen3: enable_thinking=False)")
    p.add_argument("--retriever", choices=["bow", "embedding"], default="bow")
    p.add_argument("--embedding-model", default=DEFAULT_MODEL)
    p.add_argument("--embedding-device", choices=["cpu", "cuda"], default="cpu",
                   help="where query/passage embedding runs; cpu keeps it off the vLLM GPU")
    p.add_argument("--concurrency", type=int, default=1, help="jobs in flight at once (closed loop)")
    p.add_argument("--arrival-rate", type=float, default=None, metavar="R",
                   help="open loop: jobs arrive R per second on average, independent of the system")
    p.add_argument("--arrival-cv", type=float, default=1.0,
                   help="coefficient of variation of the gaps between arrivals: 1 = Poisson, "
                        ">1 = bursty (gamma-distributed gaps with the same mean)")
    p.add_argument("--dataset-order", action="store_true",
                   help="run jobs in dataset order (2-hop first, 4-hop last) instead of shuffled")
    p.add_argument("--retrieve-delay", default=None, metavar="SPEC",
                   help="emulated search-backend latency added to each retrieve: SECONDS, "
                        "uniform:LOW,HIGH or lognormal:MEDIAN,SIGMA (see agent/delay.py)")
    p.add_argument("--retrieve-workers", type=int, default=0, metavar="K",
                   help="retrieves that can run at once (search + delay hold a worker); "
                        "0 = unlimited. Fewer than --concurrency makes jobs queue for retrieval")
    p.add_argument("--llm-slots", type=int, default=0, metavar="M",
                   help="LLM calls in flight at once, queued in this client (FCFS); 0 = unlimited")
    p.add_argument("--no-stream", action="store_true",
                   help="plain (non-streamed) vLLM replies: no ttft_s/decode_s in the traces")
    p.add_argument("--metrics-interval", type=float, default=0.5,
                   help="seconds between vLLM /metrics samples; 0 turns sampling off")
    p.add_argument("--run-id", default=None,
                   help="exact run folder name; default: <prefix>_<settings label> (see run_label)")
    p.add_argument("--run-prefix", default=None, help="start of the default run folder name; "
                   "default: a timestamp (the sbatch script passes run-<slurm job id>)")
    p.add_argument("--out-dir", default="results/traces")
    args = p.parse_args()
    if args.concurrency < 1:
        p.error("--concurrency must be >= 1")
    if args.arrival_rate is not None and (args.arrival_rate <= 0 or args.concurrency != 1):
        p.error("--arrival-rate must be > 0, and replaces --concurrency (open vs closed loop)")
    if args.arrival_cv <= 0:
        p.error("--arrival-cv must be > 0")
    if args.retrieve_workers < 0 or args.llm_slots < 0:
        p.error("--retrieve-workers and --llm-slots must be >= 0")
    try:
        delay_sampler = parse_delay(args.retrieve_delay) if args.retrieve_delay else None
    except ValueError as e:
        p.error(str(e))

    workload = get_workload("agentic_rag_tracer", args.data_dir)
    pool = workload.reduced_slice(seed=args.seed) if args.tasks == "slice" else list(workload.tasks())
    tasks = pool
    if args.limit is not None and args.limit < len(tasks):
        picked = {t.task_id for t in random.Random(args.seed).sample(tasks, args.limit)}
        tasks = [t for t in tasks if t.task_id in picked]
    if not args.dataset_order:
        tasks = list(tasks)
        random.Random(f"{args.seed}:order").shuffle(tasks)

    docs = pool_documents(pool)
    embedding_device = None
    if args.retriever == "embedding":
        embedder = SentenceTransformerEmbedder(args.embedding_model, device=args.embedding_device)
        embedding_device = str(embedder.model.device)
        retriever = LockedRetriever(EmbeddingRetriever.from_documents(docs, embedder))
    else:
        retriever = LocalRetriever.from_documents(docs)   # read-only after build: thread-safe

    model = None
    if args.llm == "vllm":
        model = args.model or VLLMBackend.served_model(args.base_url)
        # AgenticRAGTracer questions are lookups and comparisons: no calculator,
        # which the model otherwise calls in loops instead of answering.
        backend = VLLMBackend(model, base_url=args.base_url, max_tokens=args.max_tokens,
                              repetition_penalty=args.repetition_penalty,
                              system_prompt=react_system_prompt(("retrieve", "finish")),
                              chat_template_kwargs={"enable_thinking": False} if args.no_thinking else None,
                              stream=not args.no_stream)

    mode = "open_loop" if args.arrival_rate else "serial" if args.concurrency == 1 else "concurrent"
    run_id = args.run_id or f"{args.run_prefix or time.strftime('%Y%m%d-%H%M%S')}_{run_label(args, model, len(tasks))}"
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "mode": mode,
        "concurrency": None if args.arrival_rate else args.concurrency,
        "arrival_rate": args.arrival_rate,
        "arrival_cv": args.arrival_cv if args.arrival_rate else None,
        "job_order": "dataset" if args.dataset_order else "shuffled",
        "started_at": time.time(),
        "host": socket.gethostname(),
        "gpu": gpu_name(),
        "git_commit": git_commit(),
        "llm": args.llm,
        "model": model,
        "decoding": backend.decoding if args.llm == "vllm" else None,
        "task_set": args.tasks,
        "retrieve_delay": args.retrieve_delay,
        "retrieve_workers": args.retrieve_workers or None,
        "llm_slots": args.llm_slots or None,
        "embedding_device": embedding_device,
        # what the sbatch script launched the server with, plus any VLLM_* env overrides
        "vllm_serve_cmd": os.environ.get("PAA_VLLM_CMD"),
        "vllm_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("VLLM_")},
        "args": vars(args),
        "num_tasks": len(tasks),
        "corpus_docs": len(docs),
        "jobs": [],
    }
    manifest_path = run_dir / "manifest.json"
    lock = threading.Lock()   # guards manifest + progress counter across job threads
    slots = threading.BoundedSemaphore(args.retrieve_workers) if args.retrieve_workers else None
    llm_slots = threading.BoundedSemaphore(args.llm_slots) if args.llm_slots else None

    def save_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save_manifest()
    print(f"run {run_id}: {len(tasks)} tasks, {len(docs)} passages, llm={args.llm} model={model} "
          f"{'arrival_rate=%g/s cv=%g' % (args.arrival_rate, args.arrival_cv) if args.arrival_rate else 'concurrency=%d' % args.concurrency} retrieve_delay={args.retrieve_delay} "
          f"retrieve_workers={args.retrieve_workers or 'unlimited'} llm_slots={args.llm_slots or 'unlimited'} -> {run_dir}")

    poller = None
    if args.llm == "vllm" and args.metrics_interval > 0:
        (run_dir / "server").mkdir(exist_ok=True)
        poller = MetricsPoller(args.base_url, run_dir / "server" / "vllm_metrics.jsonl",
                               interval_s=args.metrics_interval).start()

    def run_job(task, arrival_ts: float | None = None) -> None:
        job_id = task.task_id.replace("/", "_")
        llm = backend if args.llm == "vllm" else MockLLM(mock_script(task))
        delay = job_delay(delay_sampler, args.seed, job_id) if delay_sampler else None
        t0 = time.time()
        try:
            res = run_one(task, run_dir / f"{job_id}.jsonl", workload=workload, job_id=job_id,
                          llm=llm, retriever=retriever, max_steps=args.max_steps, retrieve_delay=delay,
                          retrieve_slots=slots, llm_slots=llm_slots)
            entry = {k: res[k] for k in ("job_id", "success", "correct", "answer")}
        except Exception as e:
            entry = {"job_id": job_id, "error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
        t1 = time.time()
        entry.update(start_ts=t0, end_ts=t1, wall_s=round(t1 - t0, 3))
        if arrival_ts is not None:   # open loop: completion time counts from arrival
            entry.update(arrival_ts=arrival_ts, jct_s=round(t1 - arrival_ts, 3))
        with lock:
            manifest["jobs"].append(entry)
            save_manifest()
            n = len(manifest["jobs"])
        status = entry.get("error") or f"success={entry['success']} correct={entry['correct']}"
        print(f"[{n}/{len(tasks)}] {job_id} {entry['wall_s']:.1f}s {status}", flush=True)

    try:
        if args.arrival_rate:
            # gamma gaps with mean 1/R and the given CV (shape 1/CV^2); CV 1 = exponential = Poisson
            rng, shape = random.Random(f"{args.seed}:arrivals"), 1 / args.arrival_cv ** 2
            t0, offset = time.time(), 0.0
            with ThreadPoolExecutor(max_workers=len(tasks)) as pool_:   # a thread per job: no cap
                futures = []
                for task in tasks:
                    time.sleep(max(0.0, t0 + offset - time.time()))
                    futures.append(pool_.submit(run_job, task, t0 + offset))
                    offset += rng.gammavariate(shape, 1 / (args.arrival_rate * shape))
                for f in futures:
                    f.result()
        elif args.concurrency == 1:
            for task in tasks:
                run_job(task)
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool_:
                list(pool_.map(run_job, tasks))   # run_job catches job errors itself
    finally:
        if poller:
            poller.stop()
            manifest["metrics_samples"] = poller.samples
            manifest["metrics_errors"] = poller.errors

    jobs = manifest["jobs"]
    manifest["finished_at"] = time.time()
    manifest["num_succeeded"] = sum(1 for j in jobs if j.get("success"))
    manifest["num_correct"] = sum(1 for j in jobs if j.get("correct"))
    manifest["num_errors"] = sum(1 for j in jobs if "error" in j)
    manifest["makespan_s"] = manifest["finished_at"] - manifest["started_at"]
    manifest["jobs_per_s"] = len(jobs) / manifest["makespan_s"] if manifest["makespan_s"] > 0 else None
    if args.arrival_rate:
        jct = sorted(j["jct_s"] for j in jobs)
        manifest["jct_mean_s"] = sum(jct) / len(jct)
        manifest["jct_p95_s"] = jct[min(len(jct) - 1, int(0.95 * len(jct)))]
    save_manifest()
    print(f"done: {manifest['num_succeeded']}/{len(jobs)} finished, {manifest['num_correct']} correct, "
          f"{manifest['num_errors']} errors")


if __name__ == "__main__":
    main()
