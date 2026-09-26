"""Download raw workload data into ./data (gitignored). The only step that needs network.

    python scripts/fetch_workloads.py [agentic_rag_tracer hotpotqa musique]

HotpotQA is published on HuggingFace as parquet, so it needs `pip install -e ".[data]"`
(pyarrow); it is converted back to the original hotpot_dev_distractor_v1.json layout.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def _get(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def agentic_rag_tracer() -> None:
    for hops in (2, 3, 4):
        for kind in ("inference", "comparison"):
            name = f"{hops}hop_{kind}.jsonl"
            _get(HF.format(repo="YqjMartin/AgenticRAGTracer", path=name), DATA / "agentic_rag_tracer" / name)


def musique() -> None:
    name = "musique_ans_v1.0_dev.jsonl"
    _get(HF.format(repo="dgslibisey/MuSiQue", path=name), DATA / "musique" / name)


def hotpotqa() -> None:
    import pyarrow.parquet as pq

    tmp = _get(
        HF.format(repo="hotpotqa/hotpot_qa", path="distractor/validation-00000-of-00001.parquet"),
        DATA / "hotpotqa" / "validation.parquet",
    )
    records = []
    for r in pq.read_table(tmp).to_pylist():
        records.append({
            "_id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "type": r["type"],
            "level": r["level"],
            "supporting_facts": [[t, i] for t, i in zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"])],
            "context": [[t, list(s)] for t, s in zip(r["context"]["title"], r["context"]["sentences"])],
        })
    out = DATA / "hotpotqa" / "hotpot_dev_distractor_v1.json"
    out.write_text(json.dumps(records))
    tmp.unlink()
    print(f"  wrote {len(records)} records -> {out}")


FETCHERS = {"agentic_rag_tracer": agentic_rag_tracer, "hotpotqa": hotpotqa, "musique": musique}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workloads", nargs="*", choices=list(FETCHERS))
    for name in ap.parse_args().workloads or FETCHERS:
        print(name)
        FETCHERS[name]()
