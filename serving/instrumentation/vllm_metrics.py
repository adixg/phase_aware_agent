"""Background sampler of a vLLM server's Prometheus /metrics endpoint.

Agent traces show each job's side; this shows the server's: how many requests
are running vs queued, KV-cache occupancy, prefix-cache hits and preemptions
(vLLM evicting a running request's KV cache under memory pressure). Samples go
to one JSONL file per run, on the same time.time() clock as the trace spans:

    {"ts": 1791676005.9, "metrics": {"vllm:num_requests_running": 7.0, ...}}

Values are summed over label sets (engine, model). Histogram buckets and
_created timestamps are skipped; _sum/_count are kept, so means between samples
can be derived.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

# Substrings of the metric names worth keeping; names drift across vLLM versions
# (gpu_cache_usage_perc became kv_cache_usage_perc), so match loosely.
KEEP = ("num_requests_running", "num_requests_waiting", "cache_usage", "prefix_cache",
        "num_preemptions", "prompt_tokens_total", "generation_tokens_total",
        "request_queue_time_seconds", "time_to_first_token_seconds")


def parse_metrics(text: str, keep=KEEP) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name.endswith(("_bucket", "_created")) or not any(k in name for k in keep):
            continue
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        out[name] = out.get(name, 0.0) + value
    return out


class MetricsPoller:
    """Poll `<server>/metrics` every `interval_s` on a daemon thread until stop()."""

    def __init__(self, base_url: str, out_path: str | Path, *, interval_s: float = 0.5):
        root = base_url.rstrip("/")
        self.url = (root[:-3] if root.endswith("/v1") else root) + "/metrics"
        self.out_path = Path(out_path)
        self.interval_s = interval_s
        self.samples = 0
        self.errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vllm-metrics", daemon=True)

    def start(self) -> "MetricsPoller":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        with self.out_path.open("w") as f:
            while True:
                ts = time.time()
                try:
                    with urllib.request.urlopen(self.url, timeout=5) as r:
                        metrics = parse_metrics(r.read().decode())
                    f.write(json.dumps({"ts": ts, "metrics": metrics}) + "\n")
                    f.flush()
                    self.samples += 1
                except OSError:
                    self.errors += 1   # a missed sample is not worth failing the run over
                if self._stop.wait(max(0.0, self.interval_s - (time.time() - ts))):
                    return
