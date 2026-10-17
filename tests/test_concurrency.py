"""Delay specs, the vLLM metrics parser/poller, and run_slice's concurrent mode (mock LLM)."""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from agent.delay import job_delay, parse_delay
from serving.instrumentation.vllm_metrics import MetricsPoller, parse_metrics
from test_workloads import _write_fixtures

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("spec", ["0.25", "uniform:0.1,0.4", "lognormal:0.3,0.8"])
def test_delay_draws_depend_only_on_seed_job_and_step(spec):
    sampler = parse_delay(spec)
    a, b = job_delay(sampler, 0, "job_a"), job_delay(sampler, 0, "job_a")
    assert [a(s) for s in range(5)] == [b(s) for s in reversed(range(5))][::-1]   # call order is irrelevant
    assert all(x >= 0 for x in (a(s) for s in range(5)))
    if spec != "0.25":
        assert [a(s) for s in range(5)] != [job_delay(sampler, 0, "job_b")(s) for s in range(5)]
        assert [a(s) for s in range(5)] != [job_delay(sampler, 1, "job_a")(s) for s in range(5)]


def test_uniform_and_lognormal_shapes():
    u = job_delay(parse_delay("uniform:0.2,0.4"), 0, "j")
    assert all(0.2 <= u(s) <= 0.4 for s in range(200))
    ln = sorted(job_delay(parse_delay("lognormal:0.3,0.8"), 0, "j")(s) for s in range(2001))
    assert 0.25 < ln[1000] < 0.35    # median ~= the given median


@pytest.mark.parametrize("spec", ["-1", "abc", "uniform:1", "uniform:0.5,0.1", "lognormal:0,1", "gamma:1,2"])
def test_bad_delay_spec_fails_up_front(spec):
    with pytest.raises(ValueError, match="bad retrieve delay"):
        parse_delay(spec)


METRICS_TEXT = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_running{engine="1",model_name="m"} 2.0
vllm:num_requests_waiting{engine="0",model_name="m"} 1.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.42
vllm:num_preemptions_total{engine="0",model_name="m"} 7.0
vllm:request_queue_time_seconds_bucket{engine="0",le="0.1",model_name="m"} 5.0
vllm:request_queue_time_seconds_sum{engine="0",model_name="m"} 1.5
vllm:iteration_tokens_total_bucket{engine="0",le="1.0",model_name="m"} 9.0
python_gc_objects_collected_total{generation="0"} 123.0
"""


def test_parse_metrics_sums_labels_and_drops_buckets():
    assert parse_metrics(METRICS_TEXT) == {
        "vllm:num_requests_running": 5.0, "vllm:num_requests_waiting": 1.0,
        "vllm:kv_cache_usage_perc": 0.42, "vllm:num_preemptions_total": 7.0,
        "vllm:request_queue_time_seconds_sum": 1.5,
    }


def test_poller_samples_until_stopped(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = METRICS_TEXT.encode() if self.path == "/metrics" else b""
            self.send_response(200 if body else 404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    out = tmp_path / "m.jsonl"
    poller = MetricsPoller(f"http://127.0.0.1:{httpd.server_port}/v1", out, interval_s=0.02).start()
    threading.Event().wait(0.2)
    poller.stop()
    httpd.shutdown()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == poller.samples >= 3 and poller.errors == 0
    assert rows[0]["metrics"]["vllm:num_requests_running"] == 5.0
    assert all(a["ts"] < b["ts"] for a, b in zip(rows, rows[1:]))


def _run_slice(tmp_path, *extra):
    (tmp_path / "data").mkdir()
    _write_fixtures(tmp_path / "data")
    subprocess.run([sys.executable, "scripts/run_slice.py", "--llm", "mock", "--tasks", "all",
                    "--data-dir", str(tmp_path / "data"), "--out-dir", str(tmp_path / "out"),
                    "--run-id", "r", *extra], cwd=ROOT, check=True, capture_output=True)
    run_dir = tmp_path / "out" / "r"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    traces = {p.stem: [json.loads(line) for line in p.read_text().splitlines()] for p in run_dir.glob("*.jsonl")}
    return manifest, traces


def test_concurrent_run_overlaps_jobs_and_records_delays(tmp_path):
    manifest, traces = _run_slice(tmp_path, "--concurrency", "3", "--retrieve-delay", "0.3")
    assert manifest["mode"] == "concurrent" and manifest["concurrency"] == 3
    assert manifest["retrieve_delay"] == "0.3" and manifest["num_errors"] == 0
    assert len(traces) == manifest["num_tasks"] == len(manifest["jobs"]) == 6
    rets = [r for rows in traces.values() for r in rows if r.get("phase") == "retrieve"]
    assert rets and all(r["meta"]["injected_delay_s"] == 0.3 and r["duration_s"] >= 0.3 for r in rets)
    # 3 workers: job-seconds per second of run time well above 1 (serial would be <= 1)
    assert sum(j["wall_s"] for j in manifest["jobs"]) / manifest["makespan_s"] > 2
    spans = sorted((s["start_ts"], s["end_ts"]) for s in (rows[-1] for rows in traces.values()))
    assert any(b[0] < a[1] for a, b in zip(spans, spans[1:]))   # some jobs overlap in time


def test_concurrency_one_is_serial(tmp_path):
    manifest, traces = _run_slice(tmp_path)
    assert manifest["mode"] == "serial" and manifest["retrieve_delay"] is None
    ends = sorted((rows[-1]["start_ts"], rows[-1]["end_ts"]) for rows in traces.values())
    assert all(b[0] >= a[1] for a, b in zip(ends, ends[1:]))
    assert all("injected_delay_s" not in r["meta"] for rows in traces.values() for r in rows[:-1])


def test_retrieve_workers_cap_overlapping_retrieves(tmp_path):
    manifest, traces = _run_slice(tmp_path, "--concurrency", "3", "--retrieve-delay", "0.2",
                                  "--retrieve-workers", "1")
    assert manifest["retrieve_workers"] == 1 and manifest["num_errors"] == 0
    spans = [r for rows in traces.values() for r in rows[:-1]]
    rets = sorted((r["start_ts"], r["end_ts"]) for r in spans if r["phase"] == "retrieve")
    waits = [r for r in spans if r["phase"] == "wait"]
    assert all(b[0] >= a[1] for a, b in zip(rets, rets[1:]))   # one worker: retrieves never overlap
    assert len(waits) == len(rets) and all(w["meta"]["resource"] == "retrieve_slot" for w in waits)
    assert max(w["duration_s"] for w in waits) > 0.1           # 3 jobs on 1 worker: someone queued


def test_llm_slots_cap_overlapping_generations(tmp_path):
    manifest, traces = _run_slice(tmp_path, "--concurrency", "3", "--llm-slots", "1")
    assert manifest["llm_slots"] == 1 and manifest["num_errors"] == 0
    spans = [r for rows in traces.values() for r in rows[:-1]]
    gens = sorted((r["start_ts"], r["end_ts"]) for r in spans if r["phase"] == "llm_generate")
    waits = [r for r in spans if r["phase"] == "wait"]
    assert all(b[0] >= a[1] for a, b in zip(gens, gens[1:]))   # one slot: generations never overlap
    assert len(waits) == len(gens) and {w["meta"]["resource"] for w in waits} == {"llm_slot"}
