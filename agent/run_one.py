"""CLI: run one traced ReAct+RAG job end-to-end and write its trace.

    python -m agent.run_one --workload calculator --index 0

This is the unit the week-5 milestone scales to 50-100 of, and the unit the
concurrency sweep later launches many of at once against vLLM. Get this one
job's trace shape right first.

Until the vLLM backend lands, the LLM is a `MockLLM` with a scripted
trajectory: for `calculator` it solves the task; for every other workload it
retrieves once and answers "unknown", so those traces exercise the plumbing
but their `correct` field is meaningless.
"""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

from agent.graph import LangGraphReActAgent
from agent.llm_backend import LLMBackend, MockLLM
from agent.retriever import LocalRetriever
from agent.workloads import WORKLOADS, Task, get_workload, pool_documents
from serving.instrumentation.tracer import Tracer


def mock_script(task: Task) -> list[str]:
    if task.workload == "calculator":
        expr = task.meta["expression"]
        return [
            f"Thought: I should compute this.\nAction: calculator[{expr}]",
            f"Thought: I have the result.\nAction: finish[{task.gold_answers[0]}]",
        ]
    return [
        f"Thought: I should look this up.\nAction: retrieve[{task.question}]",
        "Thought: I cannot answer from this.\nAction: finish[unknown]",
    ]


def run_one(
    task: Task,
    out_path: str | Path,
    *,
    workload,
    job_id: str | None = None,
    llm: LLMBackend | None = None,
) -> dict:
    job_id = job_id or task.task_id
    retriever = LocalRetriever.from_documents(pool_documents([task]))
    agent = LangGraphReActAgent(llm or MockLLM(mock_script(task)), retriever)

    tracer = Tracer(job_id)
    result = agent.run(job_id, task.question, tracer)
    correct = workload.is_correct(task, result.answer)
    tracer.write(
        out_path,
        success=result.success,
        extra={
            "workload": task.workload,
            "task_id": task.task_id,
            "question": task.question,
            "answer": result.answer,
            "gold_answers": list(task.gold_answers),
            "correct": correct,
        },
    )
    return {"job_id": job_id, "task_id": task.task_id, "success": result.success,
            "answer": result.answer, "correct": correct, "out_path": str(out_path)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workload", choices=sorted(WORKLOADS), default="calculator")
    p.add_argument("--data-dir", default=None, help="root of fetched datasets (default: ./data)")
    p.add_argument("--index", type=int, default=0, help="which task of the (seeded, limited) sample to run")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--job-id", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    workload = get_workload(args.workload, args.data_dir)
    task = next(islice(workload.tasks(limit=args.limit, seed=args.seed), args.index, None), None)
    if task is None:
        raise SystemExit(f"no task at index {args.index}")
    job_id = args.job_id or task.task_id.replace("/", "_")
    out = args.out or f"results/traces/{job_id}.jsonl"
    print(json.dumps(run_one(task, out, workload=workload, job_id=job_id), indent=2))


if __name__ == "__main__":
    main()
