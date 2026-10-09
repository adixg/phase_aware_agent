"""Run AgenticRAGTracer slice tasks one after another and trace each job.

    python scripts/run_slice.py --llm vllm --limit 20
    python scripts/run_slice.py --llm mock --limit 3      # plumbing check, no server

Builds the retriever once over the whole slice's pooled passages (so retrieval
difficulty doesn't depend on --limit), then runs each task through `run_one`.
Traces go to <out-dir>/<run-id>/<job_id>.jsonl next to a manifest.json that
records the run's settings, host and git commit. A job that raises (e.g. the
server dies) is logged in the manifest and the run continues.

Until the Tracer grows `run_id` / `arrival_ts`, the run directory is what
groups jobs that ran together.
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import subprocess
import time
import traceback
from pathlib import Path

from agent.llm_backend import MockLLM, VLLMBackend, react_system_prompt
from agent.retriever import DEFAULT_MODEL, EmbeddingRetriever, LocalRetriever, SentenceTransformerEmbedder
from agent.run_one import mock_script, run_one
from agent.workloads import get_workload, pool_documents


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="root of fetched datasets (default: ./data)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="run a seeded random subset of the slice (keeps the hop mix); default: all")
    p.add_argument("--llm", choices=["vllm", "mock"], default="vllm")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default=None, help="default: whatever the server is serving")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--no-thinking", action="store_true",
                   help="turn off a hybrid model's <think> block (Qwen3: enable_thinking=False)")
    p.add_argument("--retriever", choices=["bow", "embedding"], default="bow")
    p.add_argument("--embedding-model", default=DEFAULT_MODEL)
    p.add_argument("--run-id", default=None, help="default: serial-<timestamp>")
    p.add_argument("--out-dir", default="results/traces")
    args = p.parse_args()

    workload = get_workload("agentic_rag_tracer", args.data_dir)
    slice_tasks = workload.reduced_slice(seed=args.seed)
    tasks = slice_tasks
    if args.limit is not None and args.limit < len(tasks):
        picked = {t.task_id for t in random.Random(args.seed).sample(tasks, args.limit)}
        tasks = [t for t in tasks if t.task_id in picked]

    docs = pool_documents(slice_tasks)
    if args.retriever == "embedding":
        retriever = EmbeddingRetriever.from_documents(docs, SentenceTransformerEmbedder(args.embedding_model))
    else:
        retriever = LocalRetriever.from_documents(docs)

    model = None
    if args.llm == "vllm":
        model = args.model or VLLMBackend.served_model(args.base_url)
        # AgenticRAGTracer questions are lookups and comparisons: no calculator,
        # which the model otherwise calls in loops instead of answering.
        backend = VLLMBackend(model, base_url=args.base_url, max_tokens=args.max_tokens,
                              system_prompt=react_system_prompt(("retrieve", "finish")),
                              chat_template_kwargs={"enable_thinking": False} if args.no_thinking else None)

    run_id = args.run_id or time.strftime("serial-%Y%m%d-%H%M%S")
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "mode": "serial",
        "started_at": time.time(),
        "host": socket.gethostname(),
        "git_commit": git_commit(),
        "llm": args.llm,
        "model": model,
        "args": vars(args),
        "num_tasks": len(tasks),
        "corpus_docs": len(docs),
        "jobs": [],
    }
    manifest_path = run_dir / "manifest.json"

    def save_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save_manifest()
    print(f"run {run_id}: {len(tasks)} tasks, {len(docs)} passages, llm={args.llm} model={model} -> {run_dir}")

    for i, task in enumerate(tasks, 1):
        job_id = task.task_id.replace("/", "_")
        llm = backend if args.llm == "vllm" else MockLLM(mock_script(task))
        t0 = time.time()
        try:
            res = run_one(task, run_dir / f"{job_id}.jsonl", workload=workload, job_id=job_id,
                          llm=llm, retriever=retriever)
            entry = {k: res[k] for k in ("job_id", "success", "correct", "answer")}
        except Exception as e:
            entry = {"job_id": job_id, "error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
        entry["wall_s"] = round(time.time() - t0, 3)
        manifest["jobs"].append(entry)
        save_manifest()
        status = entry.get("error") or f"success={entry['success']} correct={entry['correct']}"
        print(f"[{i}/{len(tasks)}] {job_id} {entry['wall_s']:.1f}s {status}", flush=True)

    jobs = manifest["jobs"]
    manifest["finished_at"] = time.time()
    manifest["num_succeeded"] = sum(1 for j in jobs if j.get("success"))
    manifest["num_correct"] = sum(1 for j in jobs if j.get("correct"))
    manifest["num_errors"] = sum(1 for j in jobs if "error" in j)
    save_manifest()
    print(f"done: {manifest['num_succeeded']}/{len(jobs)} finished, {manifest['num_correct']} correct, "
          f"{manifest['num_errors']} errors")


if __name__ == "__main__":
    main()
