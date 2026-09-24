"""CLI: run one traced ReAct+RAG job end-to-end and write its trace.

    python -m agent.run_one --job-id demo-0001 --out results/traces/demo-0001.jsonl

This is the unit the week-5 milestone scales to 50-100 of, and the unit the
concurrency sweep later launches many of at once against vLLM. Get this one
job's trace shape right first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.agent import ReActAgent
from agent.llm_backend import DEFAULT_SCRIPT, MockLLM
from agent.retriever import LocalRetriever
from serving.instrumentation.tracer import Tracer

DEFAULT_QUESTION = "What is the population of the capital of France, doubled?"
DEFAULT_CORPUS_DIR = Path(__file__).resolve().parent / "retriever" / "corpus"


def run_one(job_id: str, question: str, out_path: str | Path, *, corpus_dir: Path = DEFAULT_CORPUS_DIR) -> dict:
    retriever = LocalRetriever.from_dir(corpus_dir)
    llm = MockLLM(DEFAULT_SCRIPT)
    agent = ReActAgent(llm, retriever)

    tracer = Tracer(job_id)
    result = agent.run(job_id, question, tracer)
    tracer.write(out_path, success=result.success, extra={"question": question, "answer": result.answer})
    return {"job_id": job_id, "success": result.success, "answer": result.answer, "out_path": str(out_path)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job-id", default="demo-0001")
    p.add_argument("--question", default=DEFAULT_QUESTION)
    p.add_argument("--out", default="results/traces/demo-0001.jsonl")
    args = p.parse_args()

    result = run_one(args.job_id, args.question, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
