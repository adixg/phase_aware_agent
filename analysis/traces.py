"""Load the current span/job_summary JSONL schema without changing records."""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class JobTrace:
    spans: list[dict] = field(default_factory=list)
    summary: dict | None = None


def load_traces(paths) -> dict[str, JobTrace]:
    """Group one file or an iterable of files by job_id; sort by top-level step.

    Equal steps retain input order. Missing summaries are allowed by the loader,
    but metrics needing job lifetimes require them. Unknown record types fail.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    jobs = {}
    for path in paths:
        with Path(path).open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    kind = record["record_type"]
                    if kind not in ("span", "job_summary"):
                        raise ValueError(f"unknown record_type: {kind}")
                    job = jobs.setdefault(record["job_id"], JobTrace())
                    if kind == "span":
                        for key in ("step", "phase", "start_ts", "end_ts", "duration_s"):
                            record[key]
                        job.spans.append(record)
                    else:
                        for key in ("start_ts", "end_ts", "duration_s"):
                            record[key]
                        if job.summary is not None:
                            raise ValueError("duplicate job_summary")
                        job.summary = record
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
    for job in jobs.values():
        job.spans.sort(key=lambda span: span["step"])
    return jobs
